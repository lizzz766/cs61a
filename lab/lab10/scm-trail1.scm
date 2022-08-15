(begin
  (define (f x total)
    (if (< (* x x) 50)
      (f (+ x 1) (+ total x))
      total
    )
  )
  (f 1 0)
)
(define(sum-while initial-x condition add-to-total update-x)
;      (sum-while 1         '(< (* x x) 50)    'x    '(+ x 1) )
 `(begin
  (define (f x total)
    (if ,condition
      (f ,update-x (+ total ,add-to-total))
      total
    )
  )
  (f ,initial-x 0)
 )
)
(define result (sum-while 1  '(< (* x x) 50)    'x    '(+ x 1)))
(define result2 (sum-while 1   '(< (* x x) 50)    (lambda (x) x)    (lambda (x) (+ x 1))))
