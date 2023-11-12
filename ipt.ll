; ModuleID = 'output'
source_filename = "output"

%Test = type { i32, i32 }

@0 = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1

declare i32 @puts(ptr)

declare ptr @malloc(i64)

declare void @free(ptr)

define ptr @"<global>::int_to_str"(i32 %0) {
entry:
  %num = alloca i32, align 4
  store i32 %0, ptr %num, align 4
  %out = alloca ptr, align 8
  %1 = call ptr @malloc(i64 100)
  store ptr %1, ptr %out, align 8
  %mem = alloca ptr, align 8
  %2 = load ptr, ptr %out, align 8
  store ptr %2, ptr %mem, align 8
  br label %"cond_13:4"

"cond_13:4":                                      ; preds = %"loop_13:4", %entry
  %3 = load i32, ptr %num, align 4
  %4 = icmp eq i32 %3, 0
  %5 = xor i1 %4, true
  br i1 %5, label %"loop_13:4", label %"exit_13:4"

"loop_13:4":                                      ; preds = %"cond_13:4"
  %6 = load ptr, ptr %mem, align 8
  %7 = load i32, ptr %num, align 4
  %8 = srem i32 %7, 10
  %9 = trunc i32 %8 to i8
  %10 = add i8 48, %9
  store i8 %10, ptr %6, align 1
  %11 = load i32, ptr %num, align 4
  %12 = sdiv i32 %11, 10
  store i32 %12, ptr %num, align 4
  %13 = load ptr, ptr %mem, align 8
  %14 = getelementptr i8, ptr %13, i32 1
  store ptr %14, ptr %mem, align 8
  br label %"cond_13:4"

"exit_13:4":                                      ; preds = %"cond_13:4"
  %memf = alloca ptr, align 8
  %15 = load ptr, ptr %out, align 8
  store ptr %15, ptr %memf, align 8
  %memb = alloca ptr, align 8
  %16 = load ptr, ptr %mem, align 8
  %17 = getelementptr i8, ptr %16, i32 -1
  store ptr %17, ptr %memb, align 8
  br label %"cond_22:4"

"cond_22:4":                                      ; preds = %"loop_22:4", %"exit_13:4"
  %18 = load ptr, ptr %memf, align 8
  %19 = load ptr, ptr %memb, align 8
  %20 = icmp eq ptr %18, %19
  %21 = xor i1 %20, true
  br i1 %21, label %"on_first_true_22:10", label %"on_end_and_22:10"

"loop_22:4":                                      ; preds = %"on_end_and_22:10"
  %tmp = alloca i8, align 1
  %22 = load ptr, ptr %memf, align 8
  %23 = load i8, ptr %22, align 1
  store i8 %23, ptr %tmp, align 1
  %24 = load ptr, ptr %memf, align 8
  %25 = load ptr, ptr %memb, align 8
  %26 = load i8, ptr %25, align 1
  store i8 %26, ptr %24, align 1
  %27 = load ptr, ptr %memb, align 8
  %28 = load i8, ptr %tmp, align 1
  store i8 %28, ptr %27, align 1
  %29 = load ptr, ptr %memf, align 8
  %30 = getelementptr i8, ptr %29, i32 1
  store ptr %30, ptr %memf, align 8
  %31 = load ptr, ptr %memb, align 8
  %32 = getelementptr i8, ptr %31, i32 -1
  store ptr %32, ptr %memb, align 8
  br label %"cond_22:4"

"exit_22:4":                                      ; preds = %"on_end_and_22:10"
  %33 = load ptr, ptr %mem, align 8
  store i8 0, ptr %33, align 1
  %34 = load ptr, ptr %out, align 8
  ret ptr %34

"on_first_true_22:10":                            ; preds = %"cond_22:4"
  %35 = load ptr, ptr %memf, align 8
  %36 = load ptr, ptr %memb, align 8
  %37 = getelementptr i8, ptr %36, i32 -1
  %38 = icmp eq ptr %35, %37
  %39 = xor i1 %38, true
  br label %"on_end_and_22:10"

"on_end_and_22:10":                               ; preds = %"on_first_true_22:10", %"cond_22:4"
  %40 = phi i1 [ false, %"cond_22:4" ], [ %39, %"on_first_true_22:10" ]
  br i1 %40, label %"loop_22:4", label %"exit_22:4"
}

define void @main() {
entry:
  %0 = call i32 @puts(ptr @0)
  %x = alloca i32, align 4
  store i32 10, ptr %x, align 4
  %xp = alloca ptr, align 8
  store ptr %x, ptr %xp, align 8
  %y = alloca i32, align 4
  %1 = load ptr, ptr %xp, align 8
  %2 = load i32, ptr %1, align 4
  store i32 %2, ptr %y, align 4
  %ya = alloca ptr, align 8
  %3 = load i32, ptr %y, align 4
  %4 = call ptr @"<global>::int_to_str"(i32 %3)
  store ptr %4, ptr %ya, align 8
  %5 = load ptr, ptr %ya, align 8
  %6 = call i32 @puts(ptr %5)
  ret void
}

define i32 @"std::bruh"(%Test %0) {
entry:
  %x = alloca i32, align 4
  %1 = extractvalue %Test %0, 0
  store i32 %1, ptr %x, align 4
  %2 = load i32, ptr %x, align 4
  ret i32 %2
}