// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package strings implements simple functions to manipulate UTF-8 encoded strings.
//
// For information about UTF-8 strings in Go, see https://blog.golang.org/strings.
package strings

import (
	"internal/bytealg"
	"internal/stringslite"
	"math/bits"
	"unicode"
	"unicode/utf8"
)

const maxInt = int(^uint(0) >> 1)

// explode splits s into a slice of UTF-8 strings,
// one string per Unicode character up to a maximum of n (n < 0 means no limit).
// Invalid UTF-8 bytes are sliced individually.
func explode(s string, n int) []string {
	l := utf8.RuneCountInString(s)
	if n < 0 || n > l {
		n = l
	}
	a := make([]string, n)
	for i := 0; i < n-1; i++ {
		_, size := utf8.DecodeRuneInString(s)
		a[i] = s[:size]
		s = s[size:]
	}
	if n > 0 {
		a[n-1] = s
	}
	return a
}

// Count counts the number of non-overlapping instances of substr in s.
// If substr is an empty string, Count returns 1 + the number of Unicode code points in s.
func Count(s, substr string) int {
	// special case
	if len(substr) == 0 {
		return utf8.RuneCountInString(s) + 1
	}
	if len(substr) == 1 {
		return bytealg.CountString(s, substr[0])
	}
	n := 0
	for {
		i := Index(s, substr)
		if i == -1 {
			return n
		}
		n++
		s = s[i+len(substr):]
	}
}

// Contains reports whether substr is within s.
func Contains(s, substr string) bool {
	return Index(s, substr) >= 0
}

// ContainsAny reports whether any Unicode code points in chars are within s.
func ContainsAny(s, chars string) bool {
	return IndexAny(s, chars) >= 0
}

// ContainsRune reports whether the Unicode code point r is within s.
func ContainsRune(s string, r rune) bool {
	return IndexRune(s, r) >= 0
}

// ContainsFunc reports whether any Unicode code points r within s satisfy f(r).
// It stops as soon as a call to f returns true.
func ContainsFunc(s string, f func(rune) bool) bool {
	return IndexFunc(s, f) >= 0
}

// LastIndex returns the index of the last instance of substr in s, or -1 if substr is not present in s.
func LastIndex(s, substr string) int {
	n := len(substr)
	switch {
	case n == 0:
		return len(s)
	case n == 1:
		return bytealg.LastIndexByteString(s, substr[0])
	case n == len(s):
		if substr == s {
			return 0
		}
		return -1
	case n > len(s):
		return -1
	}
	return bytealg.LastIndexRabinKarp(s, substr)
}

// IndexByte returns the index of the first instance of c in s, or -1 if c is not present in s.
func IndexByte(s string, c byte) int {
	return stringslite.IndexByte(s, c)
}

// IndexRune returns the index of the first instance of the Unicode code point
// r, or -1 if rune is not present in s.
// If r is [utf8.RuneError], it returns the first instance of any
// invalid UTF-8 byte sequence.
func IndexRune(s string, r rune) int {
	const haveFastIndex = bytealg.MaxBruteForce > 0
	switch {
	case 0 <= r && r < utf8.RuneSelf:
		return IndexByte(s, byte(r))
	case r == utf8.RuneError:
		for i, r := range s {
			if r == utf8.RuneError {
				return i
			}
		}
		return -1
	case !utf8.ValidRune(r):
		return -1
	default:
		// Search for rune r using the last byte of its UTF-8 encoded form.
		// The distribution of the last byte is more uniform compared to the
		// first byte which has a 78% chance of being [240, 243, 244].
		rs := string(r)
		last := len(rs) - 1
		i := last
		fails := 0
		for i < len(s) {
			if s[i] != rs[last] {
				o := IndexByte(s[i+1:], rs[last])
				if o < 0 {
					return -1
				}
				i += o + 1
			}
			// Step backwards comparing bytes.
			for j := 1; j < len(rs); j++ {
				if s[i-j] != rs[last-j] {
					goto next
				}
			}
			return i - last
		next:
			fails++
			i++
			if (haveFastIndex && fails > bytealg.Cutover(i)) && i < len(s) ||
				(!haveFastIndex && fails >= 4+i>>4 && i < len(s)) {
				goto fallback
			}
		}
		return -1

	fallback:
		// see comment in ../bytes/bytes.go
		if haveFastIndex {
			if j := bytealg.IndexString(s[i-last:], rs); j >= 0 {
				return i + j - last
			}
		} else {
			c0 := rs[last]
			c1 := rs[last-1]
		loop:
			for ; i < len(s); i++ {
				if s[i] == c0 && s[i-1] == c1 {
					for k := 2; k < len(rs); k++ {
						if s[i-k] != rs[last-k] {
							continue loop
						}
					}
					return i - last
				}
			}
		}
		return -1
	}
}

// IndexAny returns the index of the first instance of any Unicode code point
// from chars in s, or -1 if no Unicode code point from chars is present in s.
func IndexAny(s, chars string) int {
	if chars == "" {
		// Avoid scanning all of s.
		return -1
	}
	if len(chars) == 1 {
		// Avoid scanning all of s.
		r := rune(chars[0])
		if r >= utf8.RuneSelf {
			r = utf8.RuneError
		}
		return IndexRune(s, r)
	}
	if shouldUseASCIISet(len(s)) {
		if as, isASCII := makeASCIISet(chars); isASCII {
			for i := 0; i < len(s); i++ {
				if as.contains(s[i]) {
					return i
				}
			}
			return -1
		}
	}
	for i, c := range s {
		if IndexRune(chars, c) >= 0 {
			return i
		}
	}
	return -1
}

// LastIndexAny returns the index of the last instance of any Unicode code
// point from chars in s, or -1 if no Unicode code point from chars is
// present in s.
func LastIndexAny(s, chars string) int {
	if chars == "" {
		// Avoid scanning all of s.
		return -1
	}
	if len(s) == 1 {
		rc := rune(s[0])
		if rc >= utf8.RuneSelf {
			rc = utf8.RuneError
		}
		if IndexRune(chars, rc) >= 0 {
			return 0
		}
		return -1
	}
	if shouldUseASCIISet(len(s)) {
		if as, isASCII := makeASCIISet(chars); isASCII {
			for i := len(s) - 1; i >= 0; i-- {
				if as.contains(s[i]) {
					return i
				}
			}
			return -1
		}
	}
	if len(chars) == 1 {
		rc := rune(chars[0])
		if rc >= utf8.RuneSelf {
			rc = utf8.RuneError
		}
		for i := len(s); i > 0; {
			r, size := utf8.DecodeLastRuneInString(s[:i])
			i -= size
			if rc == r {
				return i
			}
		}
		return -1
	}
	for i := len(s); i > 0; {
		r, size := utf8.DecodeLastRuneInString(s[:i])
		i -= size
		if IndexRune(chars, r) >= 0 {
			return i
		}
	}
	return -1
}

// LastIndexByte returns the index of the last instance of c in s, or -1 if c is not present in s.
func LastIndexByte(s string, c byte) int {
	return bytealg.LastIndexByteString(s, c)
}

// Generic split: splits after each instance of sep,
// including sepSave bytes of sep in the subarrays.
func genSplit(s, sep string, sepSave, n int) []string {
	if n == 0 {
		return nil
	}
	if sep == "" {
		return explode(s, n)
	}
	if n < 0 {
		n = Count(s, sep) + 1
	}

	n = min(n, len(s)+1)
	a := make([]string, n)
	n--
	i := 0
	for i < n {
		m := Index(s, sep)
		if m < 0 {
			break
		}
		a[i] = s[:m+sepSave]
		s = s[m+len(sep):]
		i++
	}
	a[i] = s
	return a[:i+1]
}

// SplitN slices s into substrings separated by sep and returns a slice of
// the substrings between those separators.
//
// The count determines the number of substrings to return:
//   - n > 0: at most n substrings; the last substring will be the unsplit remainder;
//   - n == 0: the result is nil (zero substrings);
//   - n < 0: all substrings.
//
// Edge cases for s and sep (for example, empty strings) are handled
// as described in the documentation for [Split].
//
// To split around the first instance of a separator, see [Cut].
func SplitN(s, sep string, n int) []string { return genSplit(s, sep, 0, n) }

// SplitAfterN slices s into substrings after each instance of sep and
// returns a slice of those substrings.
//
// The count determines the number of substrings to return:
//   - n > 0: at most n substrings; the last substring will be the unsplit remainder;
//   - n == 0: the result is nil (zero substrings);
//   - n < 0: all substrings.
//
// Edge cases for s and sep (for example, empty strings) are handled
// as described in the documentation for [SplitAfter].
func SplitAfterN(s, sep string, n int) []string {
	return genSplit(s, sep, len(sep), n)
}

// Split slices s into all substrings separated by sep and returns a slice of
// the substrings between those separators.
//
// If s does not contain sep and sep is not empty, Split returns a
// slice of length 1 whose only element is s.
//
// If sep is empty, Split splits after each UTF-8 sequence. If both s
// and sep are empty, Split returns an empty slice.
//
// It is equivalent to [SplitN] with a count of -1.
//
// To split around the first instance of a separator, see [Cut].
func Split(s, sep string) []string { return genSplit(s, sep, 0, -1) }

// SplitAfter slices s into all substrings after each instance of sep and
// returns a slice of those substrings.
//
// If s does not contain sep and sep is not empty, SplitAfter returns
// a slice of length 1 whose only element is s.
//
// If sep is empty, SplitAfter splits after each UTF-8 sequence. If
// both s and sep are empty, SplitAfter returns an empty slice.
//
// It is equivalent to [SplitAfterN] with a count of -1.
func SplitAfter(s, sep string) []string {
	return genSplit(s, sep, len(sep), -1)
}

var asciiSpace = [256]uint8{'\t': 1, '\n': 1, '\v': 1, '\f': 1, '\r': 1, ' ': 1}

// Fields splits the string s around each instance of one or more consecutive white space
// characters, as defined by [unicode.IsSpace], returning a slice of substrings of s or an
// empty slice if s contains only white space. Every element of the returned slice is
// non-empty. Unlike [Split], leading and trailing runs of white space characters
// are discarded.
func Fields(s string) []string {
	// First count the fields.
	// This is an exact count if s is ASCII, otherwise it is an approximation.
	n := 0
	wasSpace := 1
	// setBits is used to track which bits are set in the bytes of s.
	setBits := uint8(0)
	for i := 0; i < len(s); i++ {
		r := s[i]
		setBits |= r
		isSpace := int(asciiSpace[r])
		n += wasSpace & ^isSpace
		wasSpace = isSpace
	}

	if setBits >= utf8.RuneSelf {
		// Some runes in the input string are not ASCII.
		return FieldsFunc(s, unicode.IsSpace)
	}
	// ASCII fast path
	a := make([]string, n)
	na := 0
	fieldStart := 0
	i := 0
	// Skip spaces in the front of the input.
	for i < len(s) && asciiSpace[s[i]] != 0 {
		i++
	}
	fieldStart = i
	for i < len(s) {
		if asciiSpace[s[i]] == 0 {
			i++
			continue
		}
		a[na] = s[fieldStart:i]
		na++
		i++
		// Skip spaces in between fields.
		for i < len(s) && asciiSpace[s[i]] != 0 {
			i++
		}
		fieldStart = i
	}
	if fieldStart < len(s) { // Last field might end at EOF.
		a[na] = s[fieldStart:]
	}
	return a
}

// FieldsFunc splits the string s at each run of Unicode code points c satisfying f(c)
// and returns an array of slices of s. If all code points in s satisfy f(c) or the
// string is empty, an empty slice is returned. Every element of the returned slice is
// non-empty. Unlike [Split], leading and trailing runs of code points satisfying f(c)
// are discarded.
//
// FieldsFunc makes no guarantees about the order in which it calls f(c)
// and assumes that f always returns the same value for a given c.
func FieldsFunc(s string, f func(rune) bool) []string {
	// A span is used to record a slice of s of the form s[start:end].
	// The start index is inclusive and the end index is exclusive.
	type span struct {
		start int
		end   int
	}
	spans := make([]span, 0, 32)

	// Find the field start and end indices.
	// Doing this in a separate pass (rather than slicing the string s
	// and collecting the result substrings right away) is significantly
	// more efficient, possibly due to cache effects.
	start := -1 // valid span start if >= 0
	for end, rune := range s {
		if f(rune) {
			if start >= 0 {
				spans = append(spans, span{start, end})
				// Set start to a negative value.
				// Note: using -1 here consistently and reproducibly
				// slows down this code by a several percent on amd64.
				start = ^start
			}
		} else {
			if start < 0 {
				start = end
			}
		}
	}

	// Last field might end at EOF.
	if start >= 0 {
		spans = append(spans, span{start, len(s)})
	}

	// Create strings from recorded field indices.
	a := make([]string, len(spans))
	for i, span := range spans {
		a[i] = s[span.start:span.end]
	}

	return a
}

// Join concatenates the elements of its first argument to create a single string. The separator
// string sep is placed between elements in the resulting string.
func Join(elems []string, sep string) string {
	switch len(elems) {
	case 0:
		return ""
	case 1:
		return elems[0]
	}

	var n int
	if len(sep) > 0 {
		if len(sep) >= maxInt/(len(elems)-1) {
			panic("strings: Join output length overflow")
		}
		n += len(sep) * (len(elems) - 1)
	}
	for _, elem := range elems {
		if len(elem) > maxInt-n {
			panic("strings: Join output length overflow")
		}
		n += len(elem)
	}

	var b Builder
	b.Grow(n)
	b.WriteString(elems[0])
	for _, s := range elems[1:] {
		b.WriteString(sep)
		b.WriteString(s)
	}
	return b.String()
}

// HasPrefix reports whether the string s begins with prefix.
func HasPrefix(s, prefix string) bool {
	return stringslite.HasPrefix(s, prefix)
}

// HasSuffix reports whether the string s ends with suffix.
func HasSuffix(s, suffix string) bool {
	return stringslite.HasSuffix(s, suffix)
}

// Map returns a copy of the string s with all its characters modified
// according to the mapping function. If mapping returns a negative value, the character is
// dropped from the string with no replacement.
func Map(mapping func(rune) rune, s string) string {
	// In the worst case, the string can grow when mapped, making
	// things unpleasant. But it's so rare we barge in assuming it's
	// fine. It could also shrink but that falls out naturally.

	// The output buffer b is initialized on demand, the first
	// time a character differs.
	var b Builder

	for i, c := range s {
		r := mapping(c)
		if r == c && c != utf8.RuneError {
			continue
		}

		var width int
		if c == utf8.RuneError {
			c, width = utf8.DecodeRuneInString(s[i:])
			if width != 1 && r == c {
				continue
			}
		} else {
			width = utf8.RuneLen(c)
		}

		b.Grow(len(s) + utf8.UTFMax)
		b.WriteString(s[:i])
		if r >= 0 {
			b.WriteRune(r)
		}

		s = s[i+width:]
		break
	}

	// Fast path for unchanged input
	if b.Cap() == 0 { // didn't call b.Grow above
		return s
	}

	for _, c := range s {
		r := mapping(c)

		if r >= 0 {
			// common case
			// Due to inlining, it is more performant to determine if WriteByte should be
			// invoked rather than always call WriteRune
			if r < utf8.RuneSelf {
				b.WriteByte(byte(r))
			} else {
				// r is not an ASCII rune.
				b.WriteRune(r)
			}
		}
	}

	return b.String()
}

// According to static analysis, spaces, dashes, zeros, equals, and tabs
// are the most commonly repeated string literal,
// often used for display on fixed-width terminal windows.
// Pre-declare constants for these for O(1) repetition in the common-case.
const (
	repeatedSpaces = "" +
		"                                                                " +
		"                                                                "
	repeatedDashes = "" +
		"----------------------------------------------------------------" +
		"----------------------------------------------------------------"
	repeatedZeroes = "" +
		"0000000000000000000000000000000000000000000000000000000000000000"
	repeatedEquals = "" +
		"================================================================" +
		"================================================================"
	repeatedTabs = "" +
		"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t" +
		"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
)

// Repeat returns a new string consisting of count copies of the string s.
//
// It panics if count is negative or if the result of (len(s) * count)
// overflows.
func Repeat(s string, count int) string {
	switch count {
	case 0:
		return ""
	case 1:
		return s
	}

	// Since we cannot return an error on overflow,
	// we should panic if the repeat will generate an overflow.
	// See golang.org/issue/16237.
	if count < 0 {
		panic("strings: negative Repeat count")
	}
	hi, lo := bits.Mul(uint(len(s)), uint(count))
