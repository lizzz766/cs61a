#1-3

environments and frames



print(print(1),print(2))
>>>1
>>>2
>>>none none





- **1-3-4**

​	operators: 

​	`	2013//10=201`

​	`	2013/10=201.3`



​	using VIM( like VSCode)

​	simulate secessions in python doc like 

​	'
​	/>>>q,r=divide_exact(2013, 10)
​	/>>>q
​	201
​	/>>>r
​	3
​	'

​	by typing

​	'python3 -m doctest ex.py'

​		if has error the test will have ***

​	

​	default value in function





- 1-3-5 Statements( Conditional)

​	

​	

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704193954634.png" alt="image-20220704193954634" style="zoom:50%;" />



`if x<0:
	return -x
 elif x>=0:
 	return x`

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704194602833.png" alt="image-20220704194602833" style="zoom:50%;" />



<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704194952648.png" alt="image-20220704194952648" style="zoom:33%;" />





Boolean Algebra

​	False: False,0,'',none



- 1-3-6 Itertation

​		while循环



- 1-4-2 Fibonacci

​	

- 1-4-3 Designing Functions



domain is the set of all inputs

range is the set of output







**Give each func exactly one job**

don't repeat yourself

define func generally





pure func







- 1-4-4 Higher Order Functio n

​	feature of language 



by expressing general methods of computation



Generalize Patterns with Arguments

different geometric shapes' areas



assert statements

​	if it is not true, an error message will be printed to the user

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704203208242.png" alt="image-20220704203208242" style="zoom:33%;" />

​		

factorizing the common part of the 3 functions

​		def area(r, shape_constant):



**Generalizing over computational Processes**

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704205305241.png" alt="image-20220704205305241" style="zoom:33%;" />



<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704205528375.png" alt="image-20220704205528375" style="zoom:25%;" />

Get a function that returns a function



`def make_adder(n):
 	def adder(k):
 		return k+n
 	return adder`


eg:
>>add_three= make adder(3)
>>add_three(4)



Attention: functions defined within other function bodies are bound to names in a *local frames*(局部变量)

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704210925170.png" alt="image-20220704210925170" style="zoom:33%;" />

Cpp中的**函数指针**就有类似的功能



Higher order functions:

​	express general methods of computation

​	remove repetition from programs

​	Separate concerns among functions

 



- 1-4-6

​	lambda function

​		square = x*x: 是一个数字

​		square = lambda x: x*x: 是一个**函数**

​		函数可以当值用



Intrinsic name:

<img src="C:\Users\Microsoft\AppData\Roaming\Typora\typora-user-images\image-20220704213008912.png" alt="image-20220704213008912" style="zoom:50%;" />
