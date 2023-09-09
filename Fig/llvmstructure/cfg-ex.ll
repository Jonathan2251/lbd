define dso_local i32 @_Z6cfg_exiii(i32 signext %a, i32 signext %b, i32 signext %n) local_unnamed_addr nounwind {
entry:
  %cmp.not23 = icmp slt i32 %n, 0
  br i1 %cmp.not23, label %cleanup, label %for.body

for.cond:                                         ; preds = %for.body
  %inc = add nuw i32 %i.026, 1
  %exitcond.not = icmp eq i32 %i.026, %n
  br i1 %exitcond.not, label %cleanup, label %for.body, !llvm.loop !2

for.body:                                         ; preds = %entry, %for.cond
  %i.026 = phi i32 [ %inc, %for.cond ], [ 0, %entry ]
  %a.addr.025 = phi i32 [ %a.addr.1, %for.cond ], [ %a, %entry ]
  %b.addr.024 = phi i32 [ %b.addr.1, %for.cond ], [ %b, %entry ]
  %cmp1 = icmp slt i32 %a.addr.025, %b.addr.024
  %sub = sext i1 %cmp1 to i32
  %b.addr.1 = add nsw i32 %b.addr.024, %sub
  %add = select i1 %cmp1, i32 %i.026, i32 0
  %a.addr.1 = add nsw i32 %add, %a.addr.025
  %cmp2 = icmp eq i32 %b.addr.1, 0
  br i1 %cmp2, label %cleanup, label %for.cond

cleanup:                                          ; preds = %for.cond, %for.body, %entry
  %b.addr.2 = phi i32 [ %b, %entry ], [ 0, %for.body ], [ %b.addr.1, %for.cond ]
  %a.addr.2 = phi i32 [ %a, %entry ], [ %a.addr.1, %for.body ], [ %a.addr.1, %for.cond ]
  %cond = icmp eq i32 %a.addr.2, 10
  %inc7 = sub i32 103, %b.addr.2
  %spec.select = select i1 %cond, i32 %inc7, i32 %a.addr.2
  %add8 = add nsw i32 %spec.select, %b.addr.2
  ret i32 %add8
}


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
