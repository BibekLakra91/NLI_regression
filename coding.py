sint=lambda: int(input())
mint=lambda: map(int,input().split())
aint=lambda: list(map(int,input().split()))

def solve():
    n=sint()
    a=input()
    b=a[::-1]
    print(a is b)

t=int(input())
for _ in range(t):
    solve()