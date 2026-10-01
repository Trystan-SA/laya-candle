//! Python's `json.dumps`, byte for byte.
//!
//! The state and any structured criterion or instruction are serialised into the token
//! sequence, so the text has to match the Python implementation exactly: `{"a": 1, "b": 2}`,
//! with the space after `:` and `,` that `serde_json` omits, and floats written the way Python's
//! `repr` writes them (`1e-05`, `100000.0`). A different string is a different tokenisation,
//! which is a different prediction.

use std::borrow::Cow;
use std::io;

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{Formatter, Serializer};

#[derive(Clone, Copy, Debug, Default)]
struct PythonFormatter {
    /// Escape every non-ASCII character as `\uXXXX`, like `ensure_ascii=True`.
    ensure_ascii: bool,
}

impl Formatter for PythonFormatter {
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(float_repr(value).as_bytes())
    }

    /// A `Value` holds no f32, but anything else serialised here gets Python's spelling too,
    /// not serde_json's: Python only knows doubles, so an f32 is written as one.
    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.write_f64(writer, f64::from(value))
    }

    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        // Python escapes everything outside printable ASCII, DEL included, as UTF-16 units.
        let mut units = [0u16; 2];
        for c in fragment.chars() {
            if (' '..='~').contains(&c) {
                writer.write_all(&[c as u8])?;
            } else {
                for u in c.encode_utf16(&mut units) {
                    write!(writer, "\\u{u:04x}")?;
                }
            }
        }
        Ok(())
    }

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

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed notation for decimal
/// exponents from -4 to 15 and in `1e-05` / `1.5e+16` notation outside it.
///
/// Only finite values reach this: `serde_json` writes the others as `null` itself.
fn float_repr(value: f64) -> String {
    // ryu would print some arbitrary number for these, silently changing the tokens.
    debug_assert!(value.is_finite(), "non-finite float {value} reached float_repr");
    let (digits, exp) = shortest_digits(value.abs());
    let sign = if value.is_sign_negative() { "-" } else { "" };

    // Python's `float_repr_style = 'short'`: fixed notation when `-4 <= exp < 16`.
    if !(-4..16).contains(&exp) {
        let (head, tail) = digits.split_at(1);
        let frac = if tail.is_empty() { String::new() } else { format!(".{tail}") };
        let esign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{head}{frac}e{esign}{:02}", exp.abs());
    }
    let point = exp + 1;
    let body = if point <= 0 {
        format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (int, frac) = digits.split_at(point as usize);
        format!("{int}.{frac}")
    };
    format!("{sign}{body}")
}

/// The shortest digits that round-trip to a finite `v >= 0`, and the decimal exponent of the
/// first. Like Python, ryu takes the candidate nearest `v` and, on an exact tie, the one ending
/// in an even digit; std's `{:e}` breaks ties upwards (`…094.25` becomes `…094.3` where Python
/// writes `…094.2`).
fn shortest_digits(v: f64) -> (String, i32) {
    if v == 0.0 {
        return ("0".into(), 0);
    }
    let mut buf = ryu::Buffer::new();
    // ryu writes `0.001`, `123.0`, `1.5e16` or `1e-7`.
    let s = buf.format_finite(v);
    let (mantissa, e) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().expect("ryu writes an integer exponent")),
        None => (s, 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    // Below 1 ryu writes `0.000ddd`, so the significant digits start after the zeros.
    let (mut digits, exp) = match int {
        "0" => {
            let sig = frac.trim_start_matches('0');
            (sig.to_string(), e - 1 - (frac.len() - sig.len()) as i32)
        }
        _ => (format!("{int}{frac}"), e + int.len() as i32 - 1),
    };
    digits.truncate(digits.trim_end_matches('0').len());
    (digits, exp)
}

fn dumps_with(value: &Value, formatter: PythonFormatter) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser).expect("serialising an in-memory Value cannot fail");
    String::from_utf8(buf).expect("serde_json only ever emits UTF-8")
}

/// Serialise a JSON value the way Python's `json.dumps(v, ensure_ascii=False)` does.
pub fn dumps(value: &Value) -> String {
    dumps_with(value, PythonFormatter { ensure_ascii: false })
}

/// Serialise a JSON value the way Python's plain `json.dumps(v)` does, with every non-ASCII
/// character escaped. The reference renders structured instructions this way.
pub fn dumps_ascii(value: &Value) -> String {
    dumps_with(value, PythonFormatter { ensure_ascii: true })
}

/// The exact text a value contributes to the sequence: a string passes through untouched,
/// anything structured is dumped as JSON.
///
/// This is how the state and every criterion reach the model; structured instructions go
/// through [`render_instructions`] instead.
pub fn render(value: &Value) -> Cow<'_, str> {
    match value {
        Value::String(s) => Cow::Borrowed(s),
        other => Cow::Owned(dumps(other)),
    }
}

/// [`render`] for a question's instructions: the reference dumps structured instructions with
/// `ensure_ascii` left on, so their non-ASCII characters reach the model escaped.
pub fn render_instructions(value: &Value) -> Cow<'_, str> {
    match value {
        Value::String(s) => Cow::Borrowed(s),
        other => Cow::Owned(dumps_ascii(other)),
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
    // The tie cases are exact doubles spelled out in full, which is the point of them.
    #[allow(clippy::excessive_precision)]
    fn floats_are_written_like_python_repr() {
        let cases = [
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (0.0001, "0.0001"),
            (0.5, "0.5"),
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (100000.0, "100000.0"),
            (1234567890123456.0, "1234567890123456.0"),
            (1e16, "1e+16"),
            (123456789012345680000.0, "1.2345678901234568e+20"),
            (-2.5e300, "-2.5e+300"),
            (0.1 + 0.2, "0.30000000000000004"),
            // Exact ties between two shortest candidates go to the even digit.
            (1638415896083094.25, "1638415896083094.2"),
            (86775706647900.125, "86775706647900.12"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
        ];
        for (v, want) in cases {
            assert_eq!(float_repr(v), want, "{v:?}");
        }
        assert_eq!(dumps(&json!({"rate": 0.00001, "n": 3})), r#"{"rate": 1e-05, "n": 3}"#);
    }

    #[test]
    fn floats_parse_to_the_nearest_double() {
        let v: Value = serde_json::from_str("123456789012345680000.0").unwrap();
        assert_eq!(dumps(&v), "1.2345678901234568e+20");
    }

    #[test]
    fn ensure_ascii_escapes_like_python() {
        assert_eq!(dumps_ascii(&json!({"ask": "remboursé?"})), r#"{"ask": "rembours\u00e9?"}"#);
        assert_eq!(dumps_ascii(&json!("😀\u{7f}\n")), r#""\ud83d\ude00\u007f\n""#);
        assert_eq!(render_instructions(&json!("remboursé?")), "remboursé?");
    }

    #[test]
    fn a_string_is_not_quoted() {
        assert_eq!(render(&json!("plain text")), "plain text");
        assert_eq!(render(&json!({"a": 1})), r#"{"a": 1}"#);
    }
}
