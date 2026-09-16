//! Date-aware deterministic scoring for temporal answers.
//!
//! # Why this exists
//!
//! `docs/measurements/m13-temporal-axis.md` closed with a directive: *"Any
//! future temporal work should fix the scorer before it fixes the reader, or
//! it will keep measuring string overlap against `The sunday before 25 May
//! 2023`"*. Token F1 ([`crate::bench::token_f1`]) compares *words*. On the
//! LoCoMo category-2 stratum — the temporal one — that means an answer of
//! `2023-06-27` against a gold of `The week before 27 June 2023` earns 0.50
//! for naming the anchor it was asked to offset *from*, and an answer of
//! `2023-05-03` against `first week of May 2023` earns 0.25 for being right.
//!
//! So the direction of the fix is the opposite of the intuition: token F1 was
//! not under-crediting correct dates, it was handing out **partial credit for
//! date-shaped near-misses**.
//!
//! # The model: closed intervals of whole days
//!
//! Every temporal expression this module resolves becomes either a
//! [`DayRange`] — a closed interval of whole days — or a [`Temporal::Duration`],
//! a span with no anchor canonicalised to days. Scoring is then arithmetic on
//! days rather than overlap on tokens, and
//!
//! ```text
//! score = |response ∩ gold| / |response|
//! ```
//!
//! is **precision against the gold interval, not F1**. That asymmetry is
//! load-bearing and must not be swapped for a symmetric measure:
//!
//! - gold `June 2023`, response `2023-06-15` → `1/1` = **1.0**. Gold's
//!   granularity is the limit of what the corpus knows; a more precise answer
//!   consistent with it is correct. Symmetric F1 over days would score this
//!   0.065 and reintroduce exactly the failure M13 found.
//! - gold `7 May 2023`, response `May 2023` → `1/31` = **0.032**. A vaguer
//!   answer than the question admits is penalised — which is why precision
//!   and not recall.
//! - vagueness cannot game it upward: `2023` against gold `June 2023` scores
//!   `30/365`.
//!
//! # The grammar is pinned, and it fails closed
//!
//! The forms below were derived by surveying every answerable gold answer in
//! both corpora. An expression the grammar does not cover returns `None`, and
//! [`temporal_score`]'s caller then keeps token F1 — the safe direction. The
//! count of `None` golds *within* the temporal stratum is reported in
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
//!   unanchored offset has no reference day in the gold string.

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

/// Score a response against a gold answer on the day line.
///
/// `None` when the gold answer is not temporal — the caller then keeps token
/// F1. See the module docs for why the interval score is precision and not F1.
pub fn temporal_score(response: &str, gold: &str) -> Option<(f64, TemporalKind)> {
    match parse_gold(gold)? {
        Temporal::Range(g) => {
            let Some(Temporal::Range(r)) = parse_response(response, Some(g.lo.year())) else {
                // Gold names a time and the answer names none: no credit, and
                // no fallback to string overlap, which is the thing being
                // fixed.
                return Some((0.0, TemporalKind::Interval));
            };
            Some((
                g.overlap(r) as f64 / r.days() as f64,
                TemporalKind::Interval,
            ))
        }
        Temporal::Duration(g) => {
            let d = match parse_response(response, None) {
                Some(Temporal::Duration(d)) => d,
                _ => return Some((0.0, TemporalKind::Duration)),
            };
            // Exact on canonical days: `three months` == `3 months` ==
            // `nearly three months` == 90, and `19 days` != `11 days`.
            Some((f64::from(u8::from(d == g)), TemporalKind::Duration))
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
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut flat = String::with_capacity(chars.len() + 8);
    let mut prev_digit = false;
    let mut i = 0;
    while i < chars.len() {
        if let Some((iso, next)) = iso_date_at(&chars, i) {
            flat.push_str(&iso);
            prev_digit = false;
            i = next;
            continue;
        }
        let c = chars[i];
        if is_dash(c) {
            let next_digit = chars.get(i + 1).is_some_and(char::is_ascii_digit);
            flat.push_str(if prev_digit && next_digit { " to " } else { " " });
            prev_digit = false;
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            flat.push(c);
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
                flat.push(' ');
            }
            flat.push(c);
            prev_digit = false;
            i += 1;
            continue;
        }
        flat.push(' ');
        prev_digit = false;
        i += 1;
    }

    flat.split_whitespace()
        .map(|w| match w.parse::<i64>() {
            Ok(value) if w.bytes().all(|b| b.is_ascii_digit()) => Tok::Num {
                value,
                digits: w.len(),
            },
            _ => Tok::Word(w.to_string()),
        })
        .collect()
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

fn shift(d: NaiveDate, days: i64) -> Option<NaiveDate> {
    d.checked_add_signed(TimeDelta::try_days(days)?)
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

#[cfg(test)]
mod tests {
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
    fn interval_score_is_precision_against_gold() {
        // A more precise answer consistent with a coarser gold is correct.
        let (s, k) = temporal_score("2023-06-15", "June 2023").unwrap();
        assert_eq!(k, TemporalKind::Interval);
        assert!((s - 1.0).abs() < 1e-12, "got {s}");
        // A vaguer answer than the question admits is penalised.
        let (s, _) = temporal_score("May 2023", "7 May 2023").unwrap();
        assert!((s - 1.0 / 31.0).abs() < 1e-12, "got {s}");
        // Vagueness cannot game it upward.
        let (s, _) = temporal_score("2023", "June 2023").unwrap();
        assert!((s - 30.0 / 365.0).abs() < 1e-12, "got {s}");
        // The headline: the anchor is not the offset.
        let (s, _) = temporal_score("2023-06-27", "The week before 27 June 2023").unwrap();
        assert_eq!(s, 0.0);
        // A gold that names a time and an answer that names none: no credit,
        // and no fallback to string overlap.
        let (s, _) = temporal_score("sometime in the spring", "7 May 2023").unwrap();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn duration_score_is_exact_on_canonical_days() {
        assert_eq!(
            temporal_score("3 months", "nearly three months"),
            Some((1.0, TemporalKind::Duration))
        );
        assert_eq!(
            temporal_score("11 days", "19 days"),
            Some((0.0, TemporalKind::Duration))
        );
        // A duration gold against a date answer earns nothing, not a fallback.
        assert_eq!(
            temporal_score("May 2023", "19 days"),
            Some((0.0, TemporalKind::Duration))
        );
    }

    #[test]
    fn non_temporal_gold_yields_none_so_the_caller_keeps_token_f1() {
        assert_eq!(temporal_score("Boston", "Boston"), None);
        assert_eq!(temporal_score("2023-05-01", "Tokyo"), None);
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
}
