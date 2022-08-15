import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字,姓名')
#type是要传入的参数的数据类型  help是该参数的提示信息
#parser.add_argument('integers', type=str, nargs='+' , help='传入的数字')#nargs是用来说明传入的参数个数，'+' 表示传入至少一个参数。
#>>>[5,4,3,2,1]
parser.add_argument('integers', type=int, nargs='+',help='传入的数字')


# parser.add_argument('param1', type=str,help='姓')
# parser.add_argument('param2', type=str,help='名')下面可用可选参数传参方式：

parser.add_argument('--family', type=str,help='姓')
parser.add_argument('--name', type=str,help='名')
#>>>python demo.py --family=张 --name=三


args = parser.parse_args()
#获得传入的参数
print(args)

#获得integers参数
print(args.integers)

print(sum(args.integers))
print(args.family+args.name)