//! `json.dumps(..., ensure_ascii=False)`, byte for byte.
//!
//! The state and any structured criterion are serialised into the token sequence, so the
//! separators have to match the Python implementation exactly: `{"a": 1, "b": 2}`, with the
//! space after `:` and `,` that `serde_json` omits. A different string is a different
//! tokenisation, which is a different prediction.

use std::borrow::Cow;
use std::io;

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{Formatter, Serializer};

#[derive(Clone, Copy, Debug, Default)]
struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Serialise a JSON value the way Python's `json.dumps(v, ensure_ascii=False)` does.
pub fn dumps(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PythonFormatter);
    value.serialize(&mut ser).expect("serialising an in-memory Value cannot fail");
    String::from_utf8(buf).expect("serde_json only ever emits UTF-8")
}

/// The exact text a value contributes to the sequence: a string passes through untouched,
/// anything structured is dumped as JSON.
///
/// This is how the state, the instructions and every criterion reach the model.
pub fn render(value: &Value) -> Cow<'_, str> {
    match value {
        Value::String(s) => Cow::Borrowed(s),
        other => Cow::Owned(dumps(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn separators_match_python() {
        let v = json!({"from": "user@acme.com", "n": 2, "tags": ["a", "b"]});
        assert_eq!(dumps(&v), r#"{"from": "user@acme.com", "n": 2, "tags": ["a", "b"]}"#);
    }

    #[test]
    fn non_ascii_is_not_escaped() {
        assert_eq!(dumps(&json!("déjà vu")), "\"déjà vu\"");
        assert_eq!(dumps(&json!("मुझसे")), "\"मुझसे\"");
    }

    #[test]
    fn a_string_is_not_quoted() {
        assert_eq!(render(&json!("plain text")), "plain text");
        assert_eq!(render(&json!({"a": 1})), r#"{"a": 1}"#);
    }
}
