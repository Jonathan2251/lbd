
/// start
define i32 @srem(i32 %b) nounwind readnone {
entry:
  %rem = srem i32 %b, 12
  ret i32 %rem
}

