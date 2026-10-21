impl<T> Option<T> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns `true` if the option is a [`Some`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_some(), true);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_some(), false);
    /// ```
    #[must_use = "if you intended to assert that this has a value, consider `.unwrap()` instead"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    pub const fn is_some(&self) -> bool {
        matches!(*self, Some(_))
    }

    /// Returns `true` if the option is a [`Some`] and the value inside of it matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_some_and(|x| x > 1), true);
    ///
    /// let x: Option<u32> = Some(0);
    /// assert_eq!(x.is_some_and(|x| x > 1), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_some_and(|x| x > 1), false);
    ///
    /// let x: Option<String> = Some("ownership".to_string());
    /// assert_eq!(x.as_ref().is_some_and(|x| x.len() > 1), true);
    /// println!("still alive {:?}", x);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "is_some_and", since = "1.70.0")]
    #[rustc_const_unstable(feature = "const_option_ops", issue = "143956")]
    pub const fn is_some_and(self, f: impl [const] FnOnce(T) -> bool + [const] Destruct) -> bool {
        match self {
            None => false,
            Some(x) => f(x),
        }
    }

    /// Returns `true` if the option is a [`None`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_none(), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_none(), true);
    /// ```
    #[must_use = "if you intended to assert that this doesn't have a value, consider \
                  wrapping this in an `assert!()` instead"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    pub const fn is_none(&self) -> bool {
        !self.is_some()
    }

    /// Returns `true` if the option is a [`None`] or the value inside of it matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_none_or(|x| x > 1), true);
    ///
    /// let x: Option<u32> = Some(0);
    /// assert_eq!(x.is_none_or(|x| x > 1), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_none_or(|x| x > 1), true);
    ///
    /// let x: Option<String> = Some("ownership".to_string());
    /// assert_eq!(x.as_ref().is_none_or(|x| x.len() > 1), true);
    /// println!("still alive {:?}", x);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "is_none_or", since = "1.82.0")]
    #[rustc_const_unstable(feature = "const_option_ops", issue = "143956")]
    pub const fn is_none_or(self, f: impl [const] FnOnce(T) -> bool + [const] Destruct) -> bool {
        match self {
            None => true,
            Some(x) => f(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `&Option<T>` to `Option<&T>`.
    ///
    /// # Examples
    ///
    /// Calculates the length of an <code>Option<[String]></code> as an <code>Option<[usize]></code>
    /// without moving the [`String`]. The [`map`] method takes the `self` argument by value,
    /// consuming the original, so this technique uses `as_ref` to first take an `Option` to a
    /// reference to the value inside the original.
    ///
    /// [`map`]: Option::map
    /// [String]: ../../std/string/struct.String.html "String"
    /// [`String`]: ../../std/string/struct.String.html "String"
    ///
    /// ```
    /// let text: Option<String> = Some("Hello, world!".to_string());
    /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
    /// // then consume *that* with `map`, leaving `text` on the stack.
    /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
    /// println!("still can print text: {text:?}");
    /// ```
    #[inline]
    #[expect(clippy::match_as_ref, reason = "implements as_ref")]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn as_ref(&self) -> Option<&T> {
        match *self {
            Some(ref x) => Some(x),
            None => None,
        }
    }

    /// Converts from `&mut Option<T>` to `Option<&mut T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Some(2);
    /// match x.as_mut() {
    ///     Some(v) => *v = 42,
    ///     None => {},
    /// }
    /// assert_eq!(x, Some(42));
    /// ```
    #[inline]
    #[expect(clippy::match_as_ref, reason = "implements as_mut")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
    pub const fn as_mut(&mut self) -> Option<&mut T> {
        match *self {
            Some(ref mut x) => Some(x),
            None => None,
        }
    }

    /// Converts from <code>[Pin]<[&]Option\<T>></code> to <code>Option<[Pin]<[&]T>></code>.
    ///
    /// [&]: reference "shared reference"
    #[inline]
    #[must_use]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_stable(feature = "const_option_ext", since = "1.84.0")]
    pub const fn as_pin_ref(self: Pin<&Self>) -> Option<Pin<&T>> {
        // FIXME(const-hack): use `map` once that is possible
        match Pin::get_ref(self).as_ref() {
            // SAFETY: `x` is guaranteed to be pinned because it comes from `self`
            // which is pinned.
            Some(x) => unsafe { Some(Pin::new_unchecked(x)) },
            None => None,
        }
    }

    /// Converts from <code>[Pin]<[&mut] Option\<T>></code> to <code>Option<[Pin]<[&mut] T>></code>.
    ///
    /// [&mut]: reference "mutable reference"
    #[inline]
    #[must_use]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_stable(feature = "const_option_ext", since = "1.84.0")]
    pub const fn as_pin_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        // SAFETY: `get_unchecked_mut` is never used to move the `Option` inside `self`.
        // `x` is guaranteed to be pinned because it comes from `self` which is pinned.
        unsafe {
            // FIXME(const-hack): use `map` once that is possible
            match Pin::get_unchecked_mut(self).as_mut() {
                Some(x) => Some(Pin::new_unchecked(x)),
                None => None,
            }
        }
    }

    #[inline]
    const fn len(&self) -> usize {
        // Using the intrinsic avoids emitting a branch to get the 0 or 1.
        let discriminant: isize = crate::intrinsics::discriminant_value(self);
        discriminant as usize
    }

    /// Returns a slice of the contained value, if any. If this is `None`, an
    /// empty slice is returned. This can be useful to have a single type of
    /// iterator over an `Option` or slice.
    ///
    /// Note: Should you have an `Option<&T>` and wish to get a slice of `T`,
    /// you can unpack it via `opt.map_or(&[], std::slice::from_ref)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(
    ///     [Some(1234).as_slice(), None.as_slice()],
    ///     [&[1234][..], &[][..]],
    /// );
    /// ```
    ///
    /// The inverse of this function is (discounting
    /// borrowing) [`[_]::first`](slice::first):
    ///
    /// ```rust
    /// for i in [Some(1234_u16), None] {
    ///     assert_eq!(i.as_ref(), i.as_slice().first());
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "option_as_slice", since = "1.75.0")]
    #[rustc_const_stable(feature = "const_option_ext", since = "1.84.0")]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: When the `Option` is `Some`, we're using the actual pointer
        // to the payload, with a length of 1, so this is equivalent to
        // `slice::from_ref`, and thus is safe.
        // When the `Option` is `None`, the length used is 0, so to be safe it
        // just needs to be aligned, which it is because `&self` is aligned and
        // the offset used is a multiple of alignment.
        //
        // Here we assume that `offset_of!` always returns an offset to an
        // in-bounds and correctly aligned position for a `T` (even if in the
        // `None` case it's just padding).
        unsafe {
            slice::from_raw_parts(
                (self as *const Self).byte_add(core::mem::offset_of!(Self, Some.0)).cast(),
                self.len(),
            )
        }
    }

    /// Returns a mutable slice of the contained value, if any. If this is
    /// `None`, an empty slice is returned. This can be useful to have a
    /// single type of iterator over an `Option` or slice.
    ///
    /// Note: Should you have an `Option<&mut T>` instead of a
    /// `&mut Option<T>`, which this method takes, you can obtain a mutable
    /// slice via `opt.map_or(&mut [], std::slice::from_mut)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(
    ///     [Some(1234).as_mut_slice(), None.as_mut_slice()],
    ///     [&mut [1234][..], &mut [][..]],
    /// );
    /// ```
    ///
    /// The result is a mutable slice of zero or one items that points into
    /// our original `Option`:
    ///
    /// ```rust
    /// let mut x = Some(1234);
    /// x.as_mut_slice()[0] += 1;
    /// assert_eq!(x, Some(1235));
    /// ```
    ///
    /// The inverse of this method (discounting borrowing)
    /// is [`[_]::first_mut`](slice::first_mut):
    ///
    /// ```rust
    /// assert_eq!(Some(123).as_mut_slice().first_mut(), Some(&mut 123))
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "option_as_slice", since = "1.75.0")]
    #[rustc_const_stable(feature = "const_option_ext", since = "1.84.0")]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: When the `Option` is `Some`, we're using the actual pointer
        // to the payload, with a length of 1, so this is equivalent to
        // `slice::from_mut`, and thus is safe.
        // When the `Option` is `None`, the length used is 0, so to be safe it
        // just needs to be aligned, which it is because `&self` is aligned and
        // the offset used is a multiple of alignment.
        //
        // In the new version, the intrinsic creates a `*const T` from a
        // mutable reference  so it is safe to cast back to a mutable pointer
        // here. As with `as_slice`, the intrinsic always returns a pointer to
        // an in-bounds and correctly aligned position for a `T` (even if in
        // the `None` case it's just padding).
        unsafe {
            slice::from_raw_parts_mut(
                (self as *mut Self).byte_add(core::mem::offset_of!(Self, Some.0)).cast(),
                self.len(),
            )
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Getting to contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns the contained [`Some`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a [`None`] with a custom panic message provided by
    /// `msg`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("value");
    /// assert_eq!(x.expect("fruits are healthy"), "value");
    /// ```
    ///
    /// ```should_panic
    /// let x: Option<&str> = None;
    /// x.expect("fruits are healthy"); // panics with `fruits are healthy`
    /// ```
    ///
    /// # Recommended Message Style
    ///
    /// We recommend that `expect` messages are used to describe the reason you
    /// _expect_ the `Option` should be `Some`.
    ///
    /// ```should_panic
    /// # let slice: &[u8] = &[];
    /// let item = slice.get(0)
    ///     .expect("slice should not be empty");
    /// ```
    ///
    /// **Hint**: If you're having trouble remembering how to phrase expect
    /// error messages remember to focus on the word "should" as in "env
    /// variable should be set by blah" or "the given binary should be available
    /// and executable by the current user".
    ///
    /// For more detail on expect message styles and the reasoning behind our
    /// recommendation please refer to the section on ["Common Message
    /// Styles"](../../std/error/index.html#common-message-styles) in the [`std::error`](../../std/error/index.html) module docs.
    #[inline]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "option_expect"]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
    pub const fn expect(self, msg: &str) -> T {
        match self {
            Some(val) => val,
            None => expect_failed(msg),
        }
    }

    /// Returns the contained [`Some`] value, consuming the `self` value.
    ///
    /// Because this function may panic, its use is generally discouraged.
    /// Panics are meant for unrecoverable errors, and
    /// [may abort the entire program][panic-abort].
    ///
    /// Instead, prefer to use pattern matching and handle the [`None`]
    /// case explicitly, or call [`unwrap_or`], [`unwrap_or_else`], or
    /// [`unwrap_or_default`]. In functions returning `Option`, you can use
    /// [the `?` (try) operator][try-option].
    ///
    /// [panic-abort]: https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html
    /// [try-option]: https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#where-the--operator-can-be-used
    /// [`unwrap_or`]: Option::unwrap_or
    /// [`unwrap_or_else`]: Option::unwrap_or_else
    /// [`unwrap_or_default`]: Option::unwrap_or_default
    ///
    /// # Panics
    ///
    /// Panics if the self value equals [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("air");
    /// assert_eq!(x.unwrap(), "air");
    /// ```
    ///
    /// ```should_panic
    /// let x: Option<&str> = None;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    #[inline(always)]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "option_unwrap"]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
    pub const fn unwrap(self) -> T {
        match self {
            Some(val) => val,
            None => unwrap_failed(),
        }
    }

    /// Returns the contained [`Some`] value or a provided default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`unwrap_or_else`]: Option::unwrap_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(Some("car").unwrap_or("bike"), "car");
    /// assert_eq!(None.unwrap_or("bike"), "bike");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_unstable(feature = "const_option_ops", issue = "143956")]
    pub const fn unwrap_or(self, default: T) -> T
    where
        T: [const] Destruct,
    {
        match self {
            Some(x) => x,
