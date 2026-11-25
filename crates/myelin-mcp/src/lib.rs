//! `myelin-mcp` as a library: the MCP server over `myelin-core`.
//!
//! The binary (`main.rs`) and the integration tests serve the same
//! [`server::MyelinServer`]; a test hands it a backend of stand-ins and
//! speaks MCP to it in-process.

pub mod server;
