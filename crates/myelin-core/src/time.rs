//! Time as arithmetic: a pinned grammar for temporal expressions, and an
//! anchored resolver the read path can call.
//!
//! # Why this lives in `myelin-core`
//!
//! The grammar below was written for the evaluation harness's *scorer* (M14,
//! `docs/measurements/m14-temporal-scorer.md`) and stayed there for four
//! milestones. M19 needs the same grammar on the **read path**: a record whose
//! text says "I joined a new group last Tuesday" is only answerable if
//! something resolves `last Tuesday` against the day the record was recorded,
//! and the only place that knows both is `compose`. A grammar that exists twice
//! is a grammar that diverges, so it moved here and the scorer imports it.
//!
//! # The model: closed intervals of whole days
//!
//! Every temporal expression this module resolves becomes either a
//! [`DayRange`] — a closed interval of whole days — or a [`Temporal::Duration`],
//! a span with no anchor canonicalised to days. Scoring is then arithmetic on
//! days rather than overlap on tokens; see [`crate::time::parse_gold`] and the
//! scorer in `myelin-eval` for what is done with the result.
//!
//! # The grammar is pinned, and it fails closed
//!
//! The forms below were derived by surveying every answerable gold answer in
//! both corpora. An expression the grammar does not cover returns `None`, and
//! the scorer's caller then keeps token F1 — the safe direction. The count of
//! `None` golds *within* the temporal stratum is reported in
//! `docs/measurements/m14-temporal-scorer.md` as the grammar's own coverage
//! limit, rather than hidden.
//!
//! Three deliberate non-handlings, each a decision and not an omission:
//!
//! - **A bare number** (`three`, `one`) is never a duration: it carries no
//!   unit, and in this corpus is as often a count.
//! - **A day or month with no year** (`13 August`) is unresolvable on the gold
//!   side. On the *response* side only, [`parse_response`] retries with the
//!   gold's year — but only for expressions that **name a month**, so `Last
//!   summer` against a 2023 gold stays `None` instead of manufacturing a
//!   correct answer out of a word that names no year.
//! - **Open-ended and unanchored relatives** (`Since 2016`, `10 years ago`)
//!   return `None`: an open interval has no comparable length, and an
//!   unanchored offset has no reference day *in the gold string*.
//!
//! That last one is what [`resolve_relative`] supplies the missing half of. In
//! a gold answer there is no anchor; in a **record** there is, and it is
//! `record.validity.t_valid`. The resolver is a separate, deliberately smaller
//! grammar over exactly the unanchored forms the gold grammar refuses, and it
//! fails closed the same way.

use chrono::{Datelike, NaiveDate, TimeDelta, Weekday};

/// A closed interval of whole days, inclusive at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DayRange {
    pub lo: NaiveDate,
    pub hi: NaiveDate,
}

impl DayRange {
    /// Endpoints in either order; the range is normalised so `lo <= hi`.
    pub fn new(a: NaiveDate, b: NaiveDate) -> Self {
        if a <= b {
            Self { lo: a, hi: b }
        } else {
            Self { lo: b, hi: a }
        }
    }

    /// A single day.
    pub fn day(d: NaiveDate) -> Self {
        Self { lo: d, hi: d }
    }

    /// Length in days. Always `>= 1`: both ends are inclusive.
    pub fn days(self) -> i64 {
        (self.hi - self.lo).num_days() + 1
    }

    /// Intersecting days, `0` when disjoint.
    pub fn overlap(self, other: Self) -> i64 {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            0
        } else {
            (hi - lo).num_days() + 1
        }
    }
}

/// What a temporal expression resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temporal {
    Range(DayRange),
    /// A span of time with no anchor, canonicalised to days.
    Duration(i64),
}

/// Which half of the grammar matched the gold answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalKind {
    Interval,
    Duration,
}

impl TemporalKind {
    /// The string written to `ScoredQuestion::temporal_kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            TemporalKind::Interval => "interval",
            TemporalKind::Duration => "duration",
        }
    }
}

/// Parse a curated gold answer.
///
/// Whole-string match: gold is short and curated, and a scan would let
/// `Tokyo, Canada in 2019` score as a year. Everything the grammar does not
/// consume must be a filler word, otherwise this is `None` — which is how
/// `10 years ago` stays unresolvable while `10 years` is a duration.
pub fn parse_gold(text: &str) -> Option<Temporal> {
    let toks = tokenize(text);
    let p = Parser { t: &toks };
    let (value, end) = p.expr(0)?;
    if p.skip(end) == toks.len() {
        Some(value)
    } else {
        None
    }
}

/// Parse a reader answer.
///
/// Scans for the first resolvable expression, because the reader writes prose
/// (`Last Saturday, May 20, 2023.`). `gold_year` supplies a year to a
/// month-or-day expression that omits one — but only on the retry pass, and
/// only for an expression that names a month, so a season or a bare year
/// cannot borrow it.
pub fn parse_response(text: &str, gold_year: Option<i32>) -> Option<Temporal> {
    let toks = tokenize(text);
    if let Some(v) = scan(&toks, false) {
        return Some(v);
    }
    let year = gold_year?;
    let mut with_year = toks;
    with_year.push(Tok::Num {
        value: i64::from(year),
        digits: 4,
    });
    scan(&with_year, true)
}

fn scan(toks: &[Tok], require_month: bool) -> Option<Temporal> {
    let p = Parser { t: toks };
    for i in 0..toks.len() {
        if let Some((value, end)) = p.expr(i) {
            if require_month && !p.span_has_month(i, end) {
                continue;
            }
            return Some(value);
        }
    }
    None
}

// ------------------------------------------------------- anchored resolution

/// One relative time reference found in a record's text, resolved against the
/// record's own date.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The phrase as it appears in the text, lowercased and
    /// whitespace-collapsed.
    pub phrase: String,
    pub range: DayRange,
    /// Where the phrase's first occurrence stands in the text, as a byte
    /// range into the original string (M64 annotates it in place).
    pub span: std::ops::Range<usize>,
}

/// The most results [`resolve_relative`] returns for one text.
///
/// Three, because the annotation rides in the reader's prompt: a record whose
/// text names five relative expressions is a monologue, and annotating all of
/// them costs more attention than it buys. The cap is on *distinct* phrases,
/// so a text that repeats "last Tuesday" four times still yields one.
const MAX_RESOLVED: usize = 3;

/// Scan `text` for relative time references and resolve each against `anchor`.
///
/// Fails closed, exactly as the gold grammar does: an expression outside the
/// closed set below is not returned rather than guessed at.
///
/// The set is every unanchored form observed in LoCoMo's 5,882 turns, and
/// nothing else:
///
/// - `yesterday` / `today` / `tomorrow`
/// - `last|past|this|next <weekday>` — `last` is the most recent such weekday
///   *strictly before* the anchor, `next` the first strictly after, `this` the
///   one inside the anchor's Mon–Sun week
/// - `last|past|this|next week` / `weekend` / `month` / `year`
/// - `<N> day|week|month|year[s] ago` (or `… back`), `N` a digit or a
///   number word
///
/// Everything else is refused, including the forms that *look* resolvable:
/// `a few weeks ago` (no count), `recently`, `earlier`, `since <X>` (open
/// ended), and `two weeks before` — whose anchor is another event, not the
/// session.
///
/// Results are in order of first occurrence, deduped by [`Resolved::phrase`],
/// and capped at three.
pub fn resolve_relative(text: &str, anchor: NaiveDate) -> Vec<Resolved> {
    let (toks, spans) = tokenize_spanned(text);
    let p = Parser { t: &toks };
    let mut out: Vec<Resolved> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let Some((range, end)) = p.relative(i, anchor) else {
            i += 1;
            continue;
        };
        let phrase = p.phrase(i, end);
        if !out.iter().any(|r| r.phrase == phrase) {
            let span = spans[i].start..spans[end.min(spans.len()) - 1].end;
            out.push(Resolved { phrase, range, span });
            if out.len() == MAX_RESOLVED {
                return out;
            }
        }
        i = end;
    }
    out
}

const MONTH_NAMES: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
];

fn month_name(d: NaiveDate) -> &'static str {
    MONTH_NAMES[d.month0() as usize]
}

/// A day range in words, the way LoCoMo's gold answers write dates: `7 May
/// 2023`, `May 2023`, `2023`, `2–8 June 2023`, `28 July – 3 August 2023`,
/// `28 December 2022 – 3 January 2023`. The ISO form `2023-06-02..2023-06-08`
/// is what the reader copied into answers the judge then marked wrong on 18
/// temporal rows of `m19_locomo_full` whose date was right (M64).
pub fn range_in_words(r: &DayRange) -> String {
    let (lo, hi) = (r.lo, r.hi);
    let day = |d: NaiveDate| format!("{} {} {}", d.day(), month_name(d), d.year());
    if lo == hi {
        return day(lo);
    }
    let last_of_month = |d: NaiveDate| (d + chrono::Duration::days(1)).month() != d.month();
    if lo.year() == hi.year() && lo.month() == 1 && lo.day() == 1 && hi.month() == 12 && hi.day() == 31 {
        return lo.year().to_string();
    }
    if lo.year() == hi.year() && lo.month() == hi.month() {
        if lo.day() == 1 && last_of_month(hi) {
            return format!("{} {}", month_name(lo), lo.year());
        }
        return format!("{}–{} {} {}", lo.day(), hi.day(), month_name(lo), lo.year());
    }
    if lo.year() == hi.year() {
        return format!("{} {} – {} {} {}", lo.day(), month_name(lo), hi.day(), month_name(hi), hi.year());
    }
    format!("{} – {}", day(lo), day(hi))
}

/// Does the question ask for an elapsed time or for the order of two events?
///
/// Deliberately generous: a false positive costs one extra evidence item,
/// while a false negative costs the whole mechanism on that question. It must
/// not match `when`, which 77.9% of LoCoMo's temporal stratum contains and
/// which would make the switch untargeted.
pub fn is_interval_question(text: &str) -> bool {
    const CUES: [&str; 12] = [
        "how long",
        "how many days",
        "how many weeks",
        "how many months",
        "how many years",
        "how much time",
        " before ",
        " after ",
        " since ",
        "which event",
        "earlier",
        "later",
    ];
    let lower = text.to_lowercase();
    CUES.iter().any(|c| lower.contains(c))
}

// ---------------------------------------------------------------- tokenizing

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    /// A run of digits with its digit count: `<y>` needs exactly four, so
    /// `07` must not pass as a year.
    Num { value: i64, digits: usize },
    Word(String),
}

/// Words skipped between adjacent grammar elements.
///
/// Without this, `It happened on the 7th of May 2023` resolves to the whole of
/// May instead of 7 May — measured, which is why the rule is here and not a
/// guess.
const FILLER: [&str; 7] = ["of", "the", "a", "an", "on", "in", "at"];

/// Hedges are stripped and ignored: `nearly three months` and `three months`
/// are the same duration, and `Around August 2022` is August 2022.
const HEDGE: [&str; 8] = [
    "about",
    "around",
    "approximately",
    "nearly",
    "almost",
    "over",
    "under",
    "roughly",
];

const MONTHS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

const WEEKDAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

const NUMBERWORDS: [&str; 12] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve",
];

/// Range connectives. `and` is here because the corpus writes both
/// `November 5-6, 2022` and `between October 19 and 24, 2023`.
const CONNECTIVES: [&str; 5] = ["to", "and", "until", "through", "thru"];

fn is_dash(c: char) -> bool {
    matches!(c, '-' | '\u{2013}' | '\u{2014}')
}

/// Ordinal suffix directly after a digit, at a token boundary: `3rd`, `13th`.
fn ordinal_suffix_len(chars: &[char], i: usize) -> Option<usize> {
    let a = *chars.get(i)?;
    let b = *chars.get(i + 1)?;
    let two: String = [a, b].iter().collect();
    if !matches!(two.as_str(), "st" | "nd" | "rd" | "th") {
        return None;
    }
    match chars.get(i + 2) {
        None => Some(2),
        Some(c) if !c.is_ascii_alphanumeric() => Some(2),
        Some(_) => None,
    }
}

/// `YYYY-MM-DD` rewritten to the ISO anchor form `<y> <m> <d>`.
///
/// Needed *before* the dash rule: `2023-07-15` is one date, not `2023 to 07 to
/// 15`.
fn iso_date_at(chars: &[char], i: usize) -> Option<(String, usize)> {
    if i > 0 && chars[i - 1].is_ascii_digit() {
        return None;
    }
    if i + 4 > chars.len() || !chars[i..i + 4].iter().all(char::is_ascii_digit) {
        return None;
    }
    let year: String = chars[i..i + 4].iter().collect();
    if !(year.starts_with("19") || year.starts_with("20")) {
        return None;
    }
    let mut j = i + 4;
    let field = |j: &mut usize| -> Option<String> {
        if !chars.get(*j).copied().is_some_and(is_dash) {
            return None;
        }
        *j += 1;
        let start = *j;
        while *j < chars.len() && chars[*j].is_ascii_digit() && *j - start < 2 {
            *j += 1;
        }
        if *j == start {
            return None;
        }
        Some(chars[start..*j].iter().collect())
    };
    let month = field(&mut j)?;
    let day = field(&mut j)?;
    if chars.get(j).is_some_and(char::is_ascii_digit) {
        return None;
    }
    Some((format!(" {year} {month} {day} "), j))
}

/// Normalisation, applied to both sides before matching.
///
/// 1. lowercase
/// 2. ISO dates to `<y> <m> <d>`, then a dash **between two digits** to
///    ` to `, so `November 5-6, 2022` is a two-day range
/// 3. a space between a digit and a letter in either order, so `3june` and
///    `24August,2023` tokenize (the corpus contains both)
/// 4. ordinal suffixes after a digit dropped: `3rd` → `3`
/// 5. every remaining non-ASCII-alphanumeric character to a space, runs
///    collapsed
///
/// Month matching is then by three-letter prefix, which absorbs the corpus's
/// `21Janury` and `Januarty 5, 2024` typos and the `Sept`/`Jan` abbreviations
/// without an edit-distance table.
fn tokenize(text: &str) -> Vec<Tok> {
    tokenize_spanned(text).0
}

/// [`tokenize`], with each token's byte range in the **original** `text`.
///
/// The one tokenizer: `tokenize` is this with the spans dropped. Every
/// character pushed into the normalised stream carries the byte span of the
/// source character it came from (an expanded ISO date carries its whole
/// source date, a dash's `to` carries the dash), so a token's span runs from
/// its first character's source start to its last's source end. That is
/// what lets a resolved phrase be annotated *where it stands* in the text
/// (M64) rather than after the whole record.
fn tokenize_spanned(text: &str) -> (Vec<Tok>, Vec<std::ops::Range<usize>>) {
    // Lower-cased characters, each with its source character's byte span.
    let mut chars: Vec<char> = Vec::with_capacity(text.len());
    let mut from: Vec<std::ops::Range<usize>> = Vec::with_capacity(text.len());
    for (b, ch) in text.char_indices() {
        for lc in ch.to_lowercase() {
            chars.push(lc);
            from.push(b..b + ch.len_utf8());
        }
    }
    let mut flat: Vec<char> = Vec::with_capacity(chars.len() + 8);
    let mut src: Vec<std::ops::Range<usize>> = Vec::with_capacity(chars.len() + 8);
    let push = |s: &str, span: std::ops::Range<usize>, flat: &mut Vec<char>, src: &mut Vec<std::ops::Range<usize>>| {
        for c in s.chars() {
            flat.push(c);
            src.push(span.clone());
        }
    };
    let mut prev_digit = false;
    let mut i = 0;
    while i < chars.len() {
        if let Some((iso, next)) = iso_date_at(&chars, i) {
            let span = from[i].start..from[next - 1].end;
            push(&iso, span, &mut flat, &mut src);
            prev_digit = false;
            i = next;
            continue;
        }
        let c = chars[i];
        if is_dash(c) {
            let next_digit = chars.get(i + 1).is_some_and(char::is_ascii_digit);
            let sep = if prev_digit && next_digit { " to " } else { " " };
            push(sep, from[i].clone(), &mut flat, &mut src);
            prev_digit = false;
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            push(&c.to_string(), from[i].clone(), &mut flat, &mut src);
            prev_digit = true;
            i += 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            if prev_digit {
                if let Some(n) = ordinal_suffix_len(&chars, i) {
                    i += n;
                    continue;
                }
                push(" ", from[i].clone(), &mut flat, &mut src);
            }
            push(&c.to_string(), from[i].clone(), &mut flat, &mut src);
            prev_digit = false;
            i += 1;
            continue;
        }
        push(" ", from[i].clone(), &mut flat, &mut src);
        prev_digit = false;
        i += 1;
    }

    let mut toks = Vec::new();
    let mut spans = Vec::new();
    let mut k = 0;
    while k < flat.len() {
        if flat[k].is_whitespace() {
            k += 1;
            continue;
        }
        let start = k;
        while k < flat.len() && !flat[k].is_whitespace() {
            k += 1;
        }
        let w: String = flat[start..k].iter().collect();
        toks.push(match w.parse::<i64>() {
            Ok(value) if w.bytes().all(|b| b.is_ascii_digit()) => Tok::Num {
                value,
                digits: w.len(),
            },
            _ => Tok::Word(w),
        });
        spans.push(src[start].start..src[k - 1].end);
    }
    (toks, spans)
}

// ------------------------------------------------------------------- parsing

struct Parser<'a> {
    t: &'a [Tok],
}

/// What a `week`/`weekend` expression is counting.
#[derive(Copy, Clone, PartialEq, Eq)]
enum WeekUnit {
    Week,
    Weekend,
}

/// The unit of an `N <unit> ago` offset.
#[derive(Copy, Clone, PartialEq, Eq)]
enum AgoUnit {
    Day,
    Week,
    Month,
    Year,
}

/// Which direction a `last|past|this|next` determiner points.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Deixis {
    Last,
    This,
    Next,
}

impl Parser<'_> {
    fn skip(&self, mut i: usize) -> usize {
        while let Some(Tok::Word(w)) = self.t.get(i) {
            if FILLER.contains(&w.as_str()) || HEDGE.contains(&w.as_str()) {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    fn span_has_month(&self, from: usize, to: usize) -> bool {
        self.t[from..to.min(self.t.len())]
            .iter()
            .any(|t| matches!(t, Tok::Word(w) if month_of(w).is_some()))
    }

    /// One of `words`, filler-skipped. Returns the index after it.
    fn any_word(&self, i: usize, words: &[&str]) -> Option<(usize, usize)> {
        let i = self.skip(i);
        match self.t.get(i)? {
            Tok::Word(w) => {
                let hit = words.iter().position(|k| *k == w.as_str())?;
                Some((hit, i + 1))
            }
            Tok::Num { .. } => None,
        }
    }

    fn keyword(&self, i: usize, word: &str) -> Option<usize> {
        self.any_word(i, &[word]).map(|(_, j)| j)
    }

    fn num(&self, i: usize) -> Option<(i64, usize, usize)> {
        let i = self.skip(i);
        match self.t.get(i)? {
            Tok::Num { value, digits } => Some((*value, *digits, i + 1)),
            Tok::Word(_) => None,
        }
    }

    fn year(&self, i: usize) -> Option<(i32, usize)> {
        let (value, digits, j) = self.num(i)?;
        if digits == 4 && ((1900..2000).contains(&value) || (2000..2100).contains(&value)) {
            Some((i32::try_from(value).ok()?, j))
        } else {
            None
        }
    }

    fn day(&self, i: usize) -> Option<(u32, usize)> {
        let (value, digits, j) = self.num(i)?;
        if digits <= 2 && (1..=31).contains(&value) {
            Some((u32::try_from(value).ok()?, j))
        } else {
            None
        }
    }

    fn month(&self, i: usize) -> Option<(u32, usize)> {
        let i = self.skip(i);
        match self.t.get(i)? {
            Tok::Word(w) => month_of(w).map(|m| (m, i + 1)),
            Tok::Num { .. } => None,
        }
    }

    fn weekday(&self, i: usize) -> Option<(Weekday, usize)> {
        let i = self.skip(i);
        match self.t.get(i)? {
            Tok::Word(w) => weekday_of(w).map(|d| (d, i + 1)),
            Tok::Num { .. } => None,
        }
    }

    /// An absolute single day: `<d> <month> <y>`, `<month> <d> <y>`, or
    /// `<y> <m> <d>`. A day expression **with no year is not an anchor**.
    fn anchor(&self, i: usize) -> Option<(NaiveDate, usize)> {
        if let Some((d, j)) = self.day(i) {
            if let Some((m, j)) = self.month(j) {
                if let Some((y, j)) = self.year(j) {
                    if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                        return Some((date, j));
                    }
                }
            }
        }
        if let Some((m, j)) = self.month(i) {
            if let Some((d, j)) = self.day(j) {
                if let Some((y, j)) = self.year(j) {
                    if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                        return Some((date, j));
                    }
                }
            }
        }
        if let Some((y, j)) = self.year(i) {
            if let Some((m, _, j)) = self.num(j) {
                if let Some((d, j)) = self.day(j) {
                    if let Ok(m) = u32::try_from(m) {
                        if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                            return Some((date, j));
                        }
                    }
                }
            }
        }
        None
    }

    /// A bare quantity: `3` or `three`.
    ///
    /// Deliberately excludes `last`, which is a *position* and not a count:
    /// without that split, `Last year` parses as `Duration(365)` and a reader
    /// answer that names no length scores as one.
    fn quantity(&self, i: usize) -> Option<(i64, usize)> {
        let i = self.skip(i);
        match self.t.get(i)? {
            Tok::Num { value, digits } if *digits <= 2 && *value >= 1 => Some((*value, i + 1)),
            Tok::Word(w) => NUMBERWORDS
                .iter()
                .position(|n| n == w)
                .map(|n| (n as i64 + 1, i + 1)),
            Tok::Num { .. } => None,
        }
    }

    /// A count prefix on an offset: a quantity, or `last` (= 1, as in
    /// `Last week before 13 October 2022`). `None` means the grammar's
    /// default of 1.
    fn count(&self, i: usize) -> Option<(i64, usize)> {
        if let Some(q) = self.quantity(i) {
            return Some(q);
        }
        self.keyword(i, "last").map(|j| (1, j))
    }

    /// The grammar, in order. First match wins.
    fn expr(&self, i: usize) -> Option<(Temporal, usize)> {
        if let Some(r) = self.day_range(i) {
            return Some(r);
        }
        if let Some(r) = self.weekday_offset(i) {
            return Some(r);
        }
        if let Some(r) = self.week_expr(i) {
            return Some(r);
        }
        if let Some(r) = self.day_offset(i) {
            return Some(r);
        }
        if let Some(r) = self.month_part(i) {
            return Some(r);
        }
        if let Some(r) = self.season(i) {
            return Some(r);
        }
        if let Some((d, j)) = self.anchor(i) {
            return Some((Temporal::Range(DayRange::day(d)), j));
        }
        if let Some((m, j)) = self.month(i) {
            if let Some((y, j)) = self.year(j) {
                return Some((Temporal::Range(whole_month(y, m)?), j));
            }
        }
        if let Some((y, j)) = self.year(i) {
            return Some((Temporal::Range(whole_year(y)?), j));
        }
        self.duration(i)
    }

    /// `[night of] [between|from]` then a two-endpoint day range.
    fn day_range(&self, i: usize) -> Option<(Temporal, usize)> {
        let i = self.keyword(i, "night").unwrap_or_else(|| self.skip(i));
        let i = self
            .any_word(i, &["between", "from"])
            .map_or_else(|| self.skip(i), |(_, j)| j);

        // `<month> <d1> to <d2> <y>` and `<month1> <d1> to <month2> <d2> <y>`
        if let Some((m1, j)) = self.month(i) {
            if let Some((d1, j)) = self.day(j) {
                if let Some((_, j)) = self.any_word(j, &CONNECTIVES) {
                    if let Some((m2, k)) = self.month(j) {
                        if let Some((d2, k)) = self.day(k) {
                            if let Some((y, k)) = self.year(k) {
                                return Some((range(y, m1, d1, y, m2, d2)?, k));
                            }
                        }
                    }
                    if let Some((d2, j)) = self.day(j) {
                        if let Some((y, j)) = self.year(j) {
                            return Some((range(y, m1, d1, y, m1, d2)?, j));
                        }
                    }
                }
            }
        }
        // `<d1> <month1> to <d2> <month2> <y>` and `<d1> to <d2> <month> <y>`
        if let Some((d1, j)) = self.day(i) {
            if let Some((m1, k)) = self.month(j) {
                if let Some((_, k)) = self.any_word(k, &CONNECTIVES) {
                    if let Some((d2, k)) = self.day(k) {
                        let (m2, k) = self.month(k).unwrap_or((m1, k));
                        if let Some((y, k)) = self.year(k) {
                            return Some((range(y, m1, d1, y, m2, d2)?, k));
                        }
                    }
                }
            }
            if let Some((_, j)) = self.any_word(j, &CONNECTIVES) {
                if let Some((d2, j)) = self.day(j) {
                    if let Some((m, j)) = self.month(j) {
                        if let Some((y, j)) = self.year(j) {
                            return Some((range(y, m, d1, y, m, d2)?, j));
                        }
                    }
                }
            }
        }
        None
    }

    /// `<weekday> (before|after) <anchor>` — the nearest such weekday strictly
    /// before or after the anchor.
    fn weekday_offset(&self, i: usize) -> Option<(Temporal, usize)> {
        let (wd, j) = self.weekday(i)?;
        let (dir, j) = self.any_word(j, &["before", "after"])?;
        let (anchor, j) = self.anchor(j)?;
        let step = if dir == 0 { -1 } else { 1 };
        let mut d = anchor;
        for _ in 0..7 {
            d = shift(d, step)?;
            if d.weekday() == wd {
                return Some((Temporal::Range(DayRange::day(d)), j));
            }
        }
        None
    }

    /// Every `week`/`weekend` form: offsets from an anchor, the week
    /// containing an anchor, and the ordinal weeks of a month.
    fn week_expr(&self, i: usize) -> Option<(Temporal, usize)> {
        let start = self.skip(i);
        // `last` parses as both — ordinal 0 and count 1 — and consumes the
        // same token either way, so which branch below fires decides its
        // meaning: `last week of August 2023` is the final seven days of the
        // month, `Last week before 13 October 2022` is one week back.
        let ordinal = self.ordinal(start);
        let count = self.count(start);
        let n = count.map_or(1, |(n, _)| n);
        let j = ordinal
            .map(|(_, j)| j)
            .or(count.map(|(_, j)| j))
            .unwrap_or(start);
        let (unit, j) = match self.any_word(j, &["week", "weeks", "weekend", "weekends"]) {
            Some((0 | 1, j)) => (WeekUnit::Week, j),
            Some((_, j)) => (WeekUnit::Weekend, j),
            None => return None,
        };

        if let Some((dir, j)) = self.any_word(j, &["before", "after"]) {
            let (anchor, j) = self.anchor(j)?;
            let forward = dir == 1;
            let r = match unit {
                WeekUnit::Week => week_offset(anchor, n, forward)?,
                WeekUnit::Weekend => weekend_offset(anchor, n, forward)?,
            };
            return Some((Temporal::Range(r), j));
        }
        // `of` is also a filler, so `skip` eats it on the way into the anchor;
        // consuming it here too would be a no-op at best and a failed match
        // at worst.
        if let Some((anchor, j)) = self.anchor(j) {
            let r = match unit {
                WeekUnit::Week => containing_week(anchor)?,
                WeekUnit::Weekend => containing_weekend(anchor)?,
            };
            return Some((Temporal::Range(r), j));
        }
        // `(first|second|third|fourth|last) (week|weekend) of <month> <y>`
        let (ord, _) = ordinal?;
        let (m, j) = self.month(j)?;
        let (y, j) = self.year(j)?;
        let r = month_slice(y, m, ord, unit)?;
        Some((Temporal::Range(r), j))
    }

    /// `first`..`fourth` as 1..4, `last` as 0 — a sentinel meaning "the final
    /// seven days", which is not the fifth week of anything.
    fn ordinal(&self, i: usize) -> Option<(u32, usize)> {
        let (hit, j) = self.any_word(i, &["first", "second", "third", "fourth", "last"])?;
        Some((if hit == 4 { 0 } else { hit as u32 + 1 }, j))
    }

    /// `few days (before|after) <anchor>` and `N days (before|after) <anchor>`.
    fn day_offset(&self, i: usize) -> Option<(Temporal, usize)> {
        let start = self.skip(i);
        let (n, j) = match self.keyword(start, "few") {
            // `few` is five days, the widest reading that still excludes the
            // anchor itself.
            Some(j) => (5, j),
            None => self.count(start)?,
        };
        let j = self.any_word(j, &["day", "days"])?.1;
        let (dir, j) = self.any_word(j, &["before", "after"])?;
        let (anchor, j) = self.anchor(j)?;
        let r = if dir == 0 {
            DayRange::new(shift(anchor, -n)?, shift(anchor, -1)?)
        } else {
            DayRange::new(shift(anchor, 1)?, shift(anchor, n)?)
        };
        Some((Temporal::Range(r), j))
    }

    /// `(early|beginning|mid|middle|late|end) [of] <month> <y>`.
    fn month_part(&self, i: usize) -> Option<(Temporal, usize)> {
        let (hit, j) = self.any_word(
            i,
            &["early", "beginning", "mid", "middle", "late", "end"],
        )?;
        let (m, j) = self.month(j)?;
        let (y, j) = self.year(j)?;
        let whole = whole_month(y, m)?;
        let r = match hit {
            0 | 1 => DayRange::new(whole.lo, NaiveDate::from_ymd_opt(y, m, 10)?),
            2 | 3 => DayRange::new(
                NaiveDate::from_ymd_opt(y, m, 11)?,
                NaiveDate::from_ymd_opt(y, m, 20)?,
            ),
            _ => DayRange::new(NaiveDate::from_ymd_opt(y, m, 21)?, whole.hi),
        };
        Some((Temporal::Range(r), j))
    }

    /// `(spring|summer|fall|autumn) [of] <y>`, and `winter [of] <y>` as Dec of
    /// `y` through Feb of `y + 1`.
    fn season(&self, i: usize) -> Option<(Temporal, usize)> {
        let (hit, j) = self.any_word(i, &["spring", "summer", "fall", "autumn", "winter"])?;
        let (y, j) = self.year(j)?;
        let r = match hit {
            0 => DayRange::new(
                NaiveDate::from_ymd_opt(y, 3, 1)?,
                NaiveDate::from_ymd_opt(y, 5, 31)?,
            ),
            1 => DayRange::new(
                NaiveDate::from_ymd_opt(y, 6, 1)?,
                NaiveDate::from_ymd_opt(y, 8, 31)?,
            ),
            2 | 3 => DayRange::new(
                NaiveDate::from_ymd_opt(y, 9, 1)?,
                NaiveDate::from_ymd_opt(y, 11, 30)?,
            ),
            _ => DayRange::new(NaiveDate::from_ymd_opt(y, 12, 1)?, whole_month(y + 1, 2)?.hi),
        };
        Some((Temporal::Range(r), j))
    }

    /// `[hedge] (<N>|<numberword>) (year|month|week|day)[s] [old]`.
    fn duration(&self, i: usize) -> Option<(Temporal, usize)> {
        let (n, j) = self.quantity(self.skip(i))?;
        let (unit, j) = self.any_word(
            j,
            &[
                "year", "years", "month", "months", "week", "weeks", "day", "days",
            ],
        )?;
        let per = match unit {
            0 | 1 => 365,
            2 | 3 => 30,
            4 | 5 => 7,
            _ => 1,
        };
        let j = self.keyword(j, "old").unwrap_or(j);
        Some((Temporal::Duration(n * per), j))
    }

    /// The resolver's grammar: one unanchored relative expression starting at
    /// `i`, resolved against `anchor`. Returns the index after the match.
    ///
    /// **No filler skipping.** The gold grammar skips `the`/`of`/`a` between
    /// elements because a curated gold answer is one expression and everything
    /// else in it is noise; a record is prose, where skipping would let
    /// `a week in Tokyo, and next … ` join two clauses into one phrase. The
    /// phrase [`resolve_relative`] reports must be a contiguous span of the
    /// text or the annotation it writes is a fiction.
    fn relative(&self, i: usize, anchor: NaiveDate) -> Option<(DayRange, usize)> {
        let Tok::Word(head) = self.t.get(i)? else {
            return self.ago(i, anchor);
        };
        match head.as_str() {
            "yesterday" => return Some((DayRange::day(shift(anchor, -1)?), i + 1)),
            "today" => return Some((DayRange::day(anchor), i + 1)),
            "tomorrow" => return Some((DayRange::day(shift(anchor, 1)?), i + 1)),
            _ => {}
        }
        let deixis = match head.as_str() {
            "last" | "past" => Deixis::Last,
            "this" => Deixis::This,
            "next" => Deixis::Next,
            _ => return self.ago(i, anchor),
        };
        let Some(Tok::Word(unit)) = self.t.get(i + 1) else {
            return None;
        };
        let end = i + 2;
        if let Some(wd) = weekday_strict(unit) {
            // `last Tuesday` is the most recent Tuesday *strictly before* the
            // anchor, so an anchor that is itself a Tuesday resolves a week
            // back rather than to itself.
            let step: i64 = match deixis {
                Deixis::Last => -1,
                Deixis::This => return Some((DayRange::day(weekday_in_week(anchor, wd)?), end)),
                Deixis::Next => 1,
            };
            let mut d = anchor;
            for _ in 0..7 {
                d = shift(d, step)?;
                if d.weekday() == wd {
                    return Some((DayRange::day(d), end));
                }
            }
            return None;
        }
        let range = match (unit.as_str(), deixis) {
            ("week", Deixis::Last) => week_offset(anchor, 1, false)?,
            ("week", Deixis::This) => containing_week(anchor)?,
            ("week", Deixis::Next) => week_offset(anchor, 1, true)?,
            ("weekend", Deixis::Last) => weekend_offset(anchor, 1, false)?,
            ("weekend", Deixis::This) => containing_weekend(anchor)?,
            ("weekend", Deixis::Next) => weekend_offset(anchor, 1, true)?,
            ("month", d) => {
                let n = match d {
                    Deixis::Last => -1,
                    Deixis::This => 0,
                    Deixis::Next => 1,
                };
                let (y, m) = add_months(anchor.year(), anchor.month(), n)?;
                whole_month(y, m)?
            }
            ("year", d) => {
                let n = match d {
                    Deixis::Last => -1,
                    Deixis::This => 0,
                    Deixis::Next => 1,
                };
                whole_year(anchor.year() + n)?
            }
            _ => return None,
        };
        Some((range, end))
    }

    /// `<N|numberword> day|week|month|year[s] (ago|back)`.
    ///
    /// The terminator is required: without it `two weeks before the wedding`
    /// would resolve against the session date, and its anchor is the wedding.
    fn ago(&self, i: usize, anchor: NaiveDate) -> Option<(DayRange, usize)> {
        let n = match self.t.get(i)? {
            Tok::Num { value, digits } if *digits <= 2 && *value >= 1 => *value,
            Tok::Word(w) => NUMBERWORDS.iter().position(|k| k == w)? as i64 + 1,
            Tok::Num { .. } => return None,
        };
        let Some(Tok::Word(unit)) = self.t.get(i + 1) else {
            return None;
        };
        let unit = match unit.as_str() {
            "day" | "days" => AgoUnit::Day,
            "week" | "weeks" => AgoUnit::Week,
            "month" | "months" => AgoUnit::Month,
            "year" | "years" => AgoUnit::Year,
            _ => return None,
        };
        let Some(Tok::Word(tail)) = self.t.get(i + 2) else {
            return None;
        };
        if tail != "ago" && tail != "back" {
            return None;
        }
        let end = i + 3;
        let range = match unit {
            AgoUnit::Day => DayRange::day(shift(anchor, -n)?),
            AgoUnit::Week => week_offset(anchor, n, false)?,
            AgoUnit::Month => {
                let (y, m) = add_months(anchor.year(), anchor.month(), -n)?;
                whole_month(y, m)?
            }
            AgoUnit::Year => whole_year(anchor.year() - i32::try_from(n).ok()?)?,
        };
        Some((range, end))
    }

    /// The matched tokens joined by single spaces — the phrase as it appears
    /// in the text, lowercased and whitespace-collapsed.
    fn phrase(&self, from: usize, to: usize) -> String {
        self.t[from..to.min(self.t.len())]
            .iter()
            .map(|t| match t {
                Tok::Word(w) => w.clone(),
                Tok::Num { value, .. } => value.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn month_of(word: &str) -> Option<u32> {
    if word.len() < 3 || !word.is_ascii() {
        return None;
    }
    MONTHS
        .iter()
        .position(|m| *m == &word[..3])
        .map(|i| u32::try_from(i).unwrap_or(0) + 1)
}

fn weekday_of(word: &str) -> Option<Weekday> {
    if word.len() < 3 || !word.is_ascii() {
        return None;
    }
    match WEEKDAYS.iter().position(|d| *d == &word[..3])? {
        0 => Some(Weekday::Mon),
        1 => Some(Weekday::Tue),
        2 => Some(Weekday::Wed),
        3 => Some(Weekday::Thu),
        4 => Some(Weekday::Fri),
        5 => Some(Weekday::Sat),
        _ => Some(Weekday::Sun),
    }
}

/// A weekday word in a *record*, where the text is prose and not a curated
/// gold answer.
///
/// Stricter than [`weekday_of`]'s three-letter prefix on purpose. That prefix
/// exists to absorb typos in short gold strings (`21Janury`), and over prose
/// it reads `month` as Monday — which resolved `last month` to the Monday
/// before the session date until this existed — and `satisfied` as Saturday.
/// Full names and the common abbreviations only.
fn weekday_strict(word: &str) -> Option<Weekday> {
    match word {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" | "tues" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" | "thur" | "thurs" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

fn shift(d: NaiveDate, days: i64) -> Option<NaiveDate> {
    d.checked_add_signed(TimeDelta::try_days(days)?)
}

/// `(y, m)` shifted by `n` whole months, either direction.
fn add_months(y: i32, m: u32, n: i64) -> Option<(i32, u32)> {
    let total = i64::from(y) * 12 + i64::from(m) - 1 + n;
    let year = i32::try_from(total.div_euclid(12)).ok()?;
    let month = u32::try_from(total.rem_euclid(12)).ok()? + 1;
    Some((year, month))
}

/// That weekday inside `d`'s Mon–Sun week.
fn weekday_in_week(d: NaiveDate, wd: Weekday) -> Option<NaiveDate> {
    let mon = shift(d, -i64::from(d.weekday().num_days_from_monday()))?;
    shift(mon, i64::from(wd.num_days_from_monday()))
}

fn whole_month(y: i32, m: u32) -> Option<DayRange> {
    let lo = NaiveDate::from_ymd_opt(y, m, 1)?;
    let next = if m == 12 {
        NaiveDate::from_ymd_opt(y + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(y, m + 1, 1)?
    };
    Some(DayRange::new(lo, next.pred_opt()?))
}

fn whole_year(y: i32) -> Option<DayRange> {
    Some(DayRange::new(
        NaiveDate::from_ymd_opt(y, 1, 1)?,
        NaiveDate::from_ymd_opt(y, 12, 31)?,
    ))
}

fn range(y1: i32, m1: u32, d1: u32, y2: i32, m2: u32, d2: u32) -> Option<Temporal> {
    Some(Temporal::Range(DayRange::new(
        NaiveDate::from_ymd_opt(y1, m1, d1)?,
        NaiveDate::from_ymd_opt(y2, m2, d2)?,
    )))
}

/// `[anchor − 7N, anchor − 7(N−1) − 1]` back, `[anchor + 7(N−1) + 1, anchor +
/// 7(N−1) + 7]` forward: seven whole days, the Nth such week away.
fn week_offset(anchor: NaiveDate, n: i64, forward: bool) -> Option<DayRange> {
    if forward {
        let base = 7 * (n - 1);
        Some(DayRange::new(
            shift(anchor, base + 1)?,
            shift(anchor, base + 7)?,
        ))
    } else {
        let base = 7 * (n - 1);
        Some(DayRange::new(
            shift(anchor, -(base + 7))?,
            shift(anchor, -(base + 1))?,
        ))
    }
}

/// The Sat+Sun of the Nth weekend in that direction.
fn weekend_offset(anchor: NaiveDate, n: i64, forward: bool) -> Option<DayRange> {
    let step: i64 = if forward { 1 } else { -1 };
    let mut sat = anchor;
    for _ in 0..7 {
        sat = shift(sat, step)?;
        if sat.weekday() == Weekday::Sat {
            break;
        }
    }
    if sat.weekday() != Weekday::Sat {
        return None;
    }
    let sat = shift(sat, step * 7 * (n - 1))?;
    Some(DayRange::new(sat, shift(sat, 1)?))
}

/// The Mon–Sun week containing `d`.
fn containing_week(d: NaiveDate) -> Option<DayRange> {
    let back = i64::from(d.weekday().num_days_from_monday());
    let mon = shift(d, -back)?;
    Some(DayRange::new(mon, shift(mon, 6)?))
}

/// The Sat+Sun of the Mon–Sun week containing `d`.
fn containing_weekend(d: NaiveDate) -> Option<DayRange> {
    let week = containing_week(d)?;
    let sat = shift(week.lo, 5)?;
    Some(DayRange::new(sat, shift(sat, 1)?))
}

/// `ord` 1..4 selects days 1–7 / 8–14 / 15–21 / 22–28; `0` means `last` — the
/// final seven days of the month, which is not the same as the fourth week.
fn month_slice(y: i32, m: u32, ord: u32, unit: WeekUnit) -> Option<DayRange> {
    let whole = whole_month(y, m)?;
    let week = if ord == 0 {
        DayRange::new(shift(whole.hi, -6)?, whole.hi)
    } else {
        let lo = NaiveDate::from_ymd_opt(y, m, (ord - 1) * 7 + 1)?;
        DayRange::new(lo, shift(lo, 6)?)
    };
    match unit {
        WeekUnit::Week => Some(week),
        // The weekend inside that seven-day window.
        WeekUnit::Weekend => {
            let mut d = week.lo;
            while d <= week.hi {
                if d.weekday() == Weekday::Sat {
                    return Some(DayRange::new(d, shift(d, 1)?));
                }
                d = shift(d, 1)?;
            }
            Some(week)
        }
    }
}

// ------------------------------------------------------------ distance to a day

/// Days in a week, for the week count in [`ago_phrase`].
const DAYS_PER_WEEK: i64 = 7;
/// A week count is stated from one full week; below that "N days ago" is
/// the whole answer and "0 weeks" would be noise.
const AGO_WEEKS_FROM_DAYS: i64 = DAYS_PER_WEEK;

/// Whole calendar months from `from` to `to` (`to >= from`), the way a
/// person counts them: 2022-10-22 → 2023-03-25 is 5 months, and 2023-01-31 →
/// 2023-02-28 is 0 because the 28th is before the 31st.
fn whole_months_between(from: NaiveDate, to: NaiveDate) -> i64 {
    let months = (i64::from(to.year()) - i64::from(from.year())) * 12
        + (i64::from(to.month()) - i64::from(from.month()));
    if to.day() < from.day() {
        months - 1
    } else {
        months
    }
}

/// How far `day` lies from `today`, stated so a duration question is a
/// lookup: `today`, `3 days ago`, `28 days ago; 4 weeks`, `154 days ago;
/// 22 weeks; 5 months`, or `in 9 days` for a day still ahead (M46).
/// Pinned by `ago_phrase_states_every_unit_a_question_might_ask_in`.
///
/// Every unit the question might ask in is stated, and every one is the
/// *floor* — "how many weeks ago" wants 4 for 30 days, not 4.3 — which is
/// also how LongMemEval's gold counts them. The day count is exclusive of
/// both ends; the corpus's gold accepts either convention ("10 days ago. 11
/// days (inclusive)") and Test of Time (`10.48550/arxiv.2406.09170`) found
/// off-by-one to be the dominant error class in model-side arithmetic, so
/// the convention is fixed here once and never left to the reader.
pub fn ago_phrase(day: NaiveDate, today: NaiveDate) -> String {
    let days = (today - day).num_days();
    if days == 0 {
        return "today".to_string();
    }
    if days < 0 {
        return format!("in {}", counted(-days, "day"));
    }
    let mut out = format!("{} ago", counted(days, "day"));
    if days >= AGO_WEEKS_FROM_DAYS {
        out.push_str(&format!("; {}", counted(days / DAYS_PER_WEEK, "week")));
    }
    let months = whole_months_between(day, today);
    if months >= 1 {
        out.push_str(&format!("; {}", counted(months, "month")));
    }
    out
}

/// `1 week`, `4 weeks`.
fn counted(n: i64, unit: &str) -> String {
    if n == 1 {
        format!("1 {unit}")
    } else {
        format!("{n} {unit}s")
    }
}

#[cfg(test)]
mod tests {
    /// M46. Each case is a LongMemEval_S question the shipped reader got
    /// wrong by doing the arithmetic itself: 28 days → "4 weeks" (declined),
    /// 154 days → "5 months" (it said 2).
    #[test]
    fn ago_phrase_states_every_unit_a_question_might_ask_in() {
        let d = |y, m, dd| chrono::NaiveDate::from_ymd_opt(y, m, dd).unwrap();
        assert_eq!(super::ago_phrase(d(2023, 3, 4), d(2023, 4, 1)), "28 days ago; 4 weeks");
        assert_eq!(
            super::ago_phrase(d(2022, 10, 22), d(2023, 3, 25)),
            "154 days ago; 22 weeks; 5 months"
        );
        assert_eq!(super::ago_phrase(d(2023, 4, 1), d(2023, 4, 1)), "today");
        assert_eq!(super::ago_phrase(d(2023, 3, 31), d(2023, 4, 1)), "1 day ago");
        assert_eq!(super::ago_phrase(d(2023, 3, 25), d(2023, 4, 1)), "7 days ago; 1 week");
        assert_eq!(super::ago_phrase(d(2023, 4, 10), d(2023, 4, 1)), "in 9 days");
        // Whole months are counted the way a person counts them: the 28th
        // is before the 31st, so no month has passed.
        assert_eq!(super::ago_phrase(d(2023, 1, 31), d(2023, 2, 28)), "28 days ago; 4 weeks");
        assert_eq!(
            super::ago_phrase(d(2023, 1, 31), d(2023, 3, 3)),
            "31 days ago; 4 weeks; 1 month"
        );
    }

    use super::*;

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    /// One assertion per grammar row. Every input is a verbatim gold answer
    /// from `runs/locomo_recall/per_question.jsonl`.
    #[track_caller]
    fn gold_range(text: &str, lo: (i32, u32, u32), hi: (i32, u32, u32)) {
        let want = DayRange::new(ymd(lo.0, lo.1, lo.2), ymd(hi.0, hi.1, hi.2));
        assert_eq!(
            parse_gold(text),
            Some(Temporal::Range(want)),
            "gold {text:?}"
        );
    }

    #[test]
    fn gold_weekday_before_anchor() {
        // 25 May 2023 is a Thursday; the Sunday before is the 21st.
        gold_range("The sunday before 25 May 2023", (2023, 5, 21), (2023, 5, 21));
        gold_range(
            "Saturday after 27 January, 2023",
            (2023, 1, 28),
            (2023, 1, 28),
        );
    }

    #[test]
    fn gold_week_offsets() {
        gold_range("The week before 3 July 2023", (2023, 6, 26), (2023, 7, 2));
        // `21Janury` — a digit glued to a misspelled month, both absorbed.
        gold_range(
            "The week before 21Janury, 2022",
            (2022, 1, 14),
            (2022, 1, 20),
        );
        gold_range("Two weeks before 11 August 2023", (2023, 7, 28), (2023, 8, 3));
        gold_range("The week of 23 August 2023", (2023, 8, 21), (2023, 8, 27));
    }

    #[test]
    fn gold_weekends() {
        gold_range("The weekend of 24June, 2022.", (2022, 6, 25), (2022, 6, 26));
        gold_range(
            "two weekends before 17 July 2023",
            (2023, 7, 8),
            (2023, 7, 9),
        );
    }

    #[test]
    fn gold_explicit_ranges() {
        gold_range(
            "On the night of October 30 to 31, 2022",
            (2022, 10, 30),
            (2022, 10, 31),
        );
        // A naive port reads the hyphen as a token separator and returns
        // `None`; normalisation rule 2 makes it a two-day range.
        gold_range("November 5-6, 2022", (2022, 11, 5), (2022, 11, 6));
        gold_range(
            "between October 19 and 24, 2023",
            (2023, 10, 19),
            (2023, 10, 24),
        );
    }

    #[test]
    fn gold_fuzzy_windows() {
        gold_range(
            "few days before November 22, 2023",
            (2023, 11, 17),
            (2023, 11, 21),
        );
        gold_range("first week of August 2023", (2023, 8, 1), (2023, 8, 7));
        gold_range("end of October 2023", (2023, 10, 21), (2023, 10, 31));
        gold_range("summer 2022", (2022, 6, 1), (2022, 8, 31));
    }

    #[test]
    fn gold_absolute_granularities() {
        gold_range("June 2023", (2023, 6, 1), (2023, 6, 30));
        gold_range("In 2013", (2013, 1, 1), (2013, 12, 31));
    }

    #[test]
    fn gold_durations_canonicalise_to_days() {
        // The hedge is stripped: `nearly three months` == `three months`.
        assert_eq!(parse_gold("nearly three months"), Some(Temporal::Duration(90)));
        assert_eq!(parse_gold("4 years"), Some(Temporal::Duration(1460)));
    }

    #[test]
    fn gold_unresolvable_returns_none() {
        // No year: unresolvable on the gold side, by design.
        assert_eq!(parse_gold("13 August"), None);
        assert_eq!(parse_gold("The week of April 3rd to 9th"), None);
        // Open-ended and unanchored relatives have no comparable length.
        assert_eq!(parse_gold("Since 2016"), None);
        assert_eq!(parse_gold("10 years ago"), None);
        // A bare number carries no unit.
        assert_eq!(parse_gold("three"), None);
        // Not temporal at all: the caller keeps token F1.
        assert_eq!(parse_gold("Tokyo"), None);
    }

    #[test]
    fn response_scans_prose_and_borrows_the_gold_year() {
        let one = |t: &str, y: i32, d: (i32, u32, u32)| {
            assert_eq!(
                parse_response(t, Some(y)),
                Some(Temporal::Range(DayRange::day(ymd(d.0, d.1, d.2)))),
                "response {t:?}"
            );
        };
        one("2023-07-15", 2023, (2023, 7, 15));
        one("Last Saturday, May 20, 2023.", 2023, (2023, 5, 20));
        // Year borrowed: the expression names a month.
        one("September 11", 2023, (2023, 9, 11));
        // The filler rule: `the 7th of May` is 7 May, not the whole month.
        one("It happened on the 7th of May 2023.", 2023, (2023, 5, 7));
    }

    #[test]
    fn response_never_borrows_a_year_for_an_expression_naming_no_month() {
        // `Last summer` genuinely does not name a year; inferring one would
        // manufacture a correct answer.
        assert_eq!(parse_response("Last summer.", Some(2023)), None);
        assert_eq!(parse_response("Last year.", Some(2022)), None);
    }

    #[test]
    fn day_range_arithmetic_is_inclusive() {
        let r = DayRange::new(ymd(2023, 1, 1), ymd(2023, 1, 7));
        assert_eq!(r.days(), 7);
        assert_eq!(DayRange::day(ymd(2023, 1, 1)).days(), 1);
        assert_eq!(r.overlap(DayRange::day(ymd(2023, 1, 7))), 1);
        assert_eq!(r.overlap(DayRange::day(ymd(2023, 1, 8))), 0);
        assert_eq!(
            r.overlap(DayRange::new(ymd(2023, 1, 5), ymd(2023, 1, 20))),
            3
        );
    }

    // ------------------------------------------------- the anchored resolver

    #[track_caller]
    fn one_resolved(text: &str, anchor: (i32, u32, u32), phrase: &str, lo: (i32, u32, u32), hi: (i32, u32, u32)) {
        let got = resolve_relative(text, ymd(anchor.0, anchor.1, anchor.2));
        assert_eq!(got.len(), 1, "resolve {text:?}: {got:?}");
        assert_eq!(got[0].phrase, phrase, "resolve {text:?}");
        assert_eq!(
            got[0].range,
            DayRange::new(ymd(lo.0, lo.1, lo.2), ymd(hi.0, hi.1, hi.2)),
            "resolve {text:?}"
        );
        // M64: the span quotes the phrase where it stands in the original text.
        let quoted: String = text[got[0].span.clone()]
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(quoted, phrase, "the span of {phrase:?} in {text:?}");
    }

    #[test]
    fn a_span_is_a_byte_range_of_the_original_text() {
        // Upper case and a non-ASCII prefix: offsets are bytes of the input,
        // not of its lower-cased copy.
        let text = "Ça va! We met LAST TUESDAY at the café.";
        let got = resolve_relative(text, ymd(2023, 7, 20));
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].span.clone()], "LAST TUESDAY");
    }

    #[test]
    fn dates_in_words_follow_the_gold_answers() {
        let r = |a: (i32, u32, u32), b: (i32, u32, u32)| {
            range_in_words(&DayRange::new(ymd(a.0, a.1, a.2), ymd(b.0, b.1, b.2)))
        };
        assert_eq!(r((2023, 5, 7), (2023, 5, 7)), "7 May 2023");
        assert_eq!(r((2023, 5, 1), (2023, 5, 31)), "May 2023");
        assert_eq!(r((2023, 1, 1), (2023, 12, 31)), "2023");
        assert_eq!(r((2023, 6, 2), (2023, 6, 8)), "2–8 June 2023");
        assert_eq!(r((2023, 7, 28), (2023, 8, 3)), "28 July – 3 August 2023");
        assert_eq!(r((2022, 12, 28), (2023, 1, 3)), "28 December 2022 – 3 January 2023");
    }

    #[test]
    fn resolve_last_weekday_is_strictly_before_the_anchor() {
        // The real LoCoMo turn behind gold "The Tuesday before 20 July 2023".
        one_resolved(
            "Hey Mel! I just joined a new LGBTQ activist group last Tuesday",
            (2023, 7, 20),
            "last tuesday",
            (2023, 7, 18),
            (2023, 7, 18),
        );
        // 18 July 2023 is itself a Tuesday: `last Tuesday` is a week back, not
        // today. An off-by-one here silently resolves every such reference to
        // the session date.
        one_resolved(
            "joined a group last Tuesday",
            (2023, 7, 18),
            "last tuesday",
            (2023, 7, 11),
            (2023, 7, 11),
        );
    }

    #[test]
    fn resolve_covers_the_closed_set() {
        one_resolved("we met yesterday", (2023, 7, 20), "yesterday", (2023, 7, 19), (2023, 7, 19));
        one_resolved("flying out tomorrow", (2023, 7, 20), "tomorrow", (2023, 7, 21), (2023, 7, 21));
        // 20 July 2023 is a Thursday, so its Mon-Sun week is 17-23 July.
        one_resolved("busy this week", (2023, 7, 20), "this week", (2023, 7, 17), (2023, 7, 23));
        one_resolved("busy last week", (2023, 7, 20), "last week", (2023, 7, 13), (2023, 7, 19));
        one_resolved("free next week", (2023, 7, 20), "next week", (2023, 7, 21), (2023, 7, 27));
        one_resolved("camped last weekend", (2023, 7, 20), "last weekend", (2023, 7, 15), (2023, 7, 16));
        // `month` shares Monday's three-letter stem, and the gold grammar
        // matches weekdays by that prefix: before `weekday_strict` existed
        // this resolved to the Monday before the session date.
        one_resolved("moved last month", (2023, 7, 20), "last month", (2023, 6, 1), (2023, 6, 30));
        one_resolved("met last Monday", (2023, 7, 20), "last monday", (2023, 7, 17), (2023, 7, 17));
        one_resolved("graduated last year", (2023, 7, 20), "last year", (2022, 1, 1), (2022, 12, 31));
        one_resolved("bought it three weeks ago", (2023, 7, 20), "three weeks ago", (2023, 6, 29), (2023, 7, 5));
        one_resolved("bought it 2 days ago", (2023, 7, 20), "2 days ago", (2023, 7, 18), (2023, 7, 18));
        one_resolved("started six months back", (2023, 7, 20), "six months back", (2023, 1, 1), (2023, 1, 31));
        // A month offset that crosses a year boundary.
        one_resolved("started eight months ago", (2023, 3, 10), "eight months ago", (2022, 7, 1), (2022, 7, 31));
    }

    #[test]
    fn resolve_fails_closed_on_everything_else() {
        let anchor = ymd(2023, 7, 20);
        // No count: `few` and `couple` are not quantities.
        assert_eq!(resolve_relative("a few years ago", anchor), vec![]);
        assert_eq!(resolve_relative("a couple of weeks ago", anchor), vec![]);
        // The anchor is another event, not the session.
        assert_eq!(resolve_relative("two weeks before", anchor), vec![]);
        assert_eq!(resolve_relative("three days after the wedding", anchor), vec![]);
        // Open-ended, and vague adverbs that name no offset.
        assert_eq!(resolve_relative("since 2016", anchor), vec![]);
        assert_eq!(resolve_relative("I saw her recently", anchor), vec![]);
        assert_eq!(resolve_relative("we talked earlier", anchor), vec![]);
        // Not a relative expression at all.
        assert_eq!(resolve_relative("on 20 July 2023 we met", anchor), vec![]);
        assert_eq!(resolve_relative("last night was rough", anchor), vec![]);
    }

    #[test]
    fn resolve_dedupes_by_phrase_and_caps_at_three() {
        let anchor = ymd(2023, 7, 20);
        let repeated = resolve_relative("last Tuesday, and again last Tuesday", anchor);
        assert_eq!(repeated.len(), 1, "{repeated:?}");
        let many = resolve_relative(
            "yesterday, today, tomorrow, last week and next month",
            anchor,
        );
        assert_eq!(
            many.iter().map(|r| r.phrase.as_str()).collect::<Vec<_>>(),
            vec!["yesterday", "today", "tomorrow"],
            "first three, in order of occurrence"
        );
    }

    #[test]
    fn interval_questions_are_recognised_without_matching_when() {
        assert!(is_interval_question(
            "How long had I been using the new area rug when I rearranged my living room furniture?"
        ));
        assert!(is_interval_question("How many weeks passed between the two visits?"));
        assert!(is_interval_question("Did I buy the rug before or after the move?"));
        assert!(is_interval_question("Which event happened earlier?"));
        // `when` alone is 77.9% of LoCoMo's temporal stratum; matching it
        // would make the switch untargeted.
        assert!(!is_interval_question(
            "When did Caroline join a new activist group?"
        ));
        assert!(!is_interval_question("What did I order for dinner?"));
    }
}
