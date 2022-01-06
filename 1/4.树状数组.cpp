//luogu P3374 【模板】树状数组 1
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 500010
using namespace std;
int n, m, op, x, y, k;
int tree[MAXN];
int lowbit(int x)
{
	return x & (-x);
}
void add(int x, int k)
{
	if (x > n)return;
	tree[x] += k;
	add(x + lowbit(x), k);
	return;
}
int sum(int x)
{
	if (x == 0)return 0;
	return tree[x] + sum(x - lowbit(x));
}
int main()
{
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)tree[i] = 0;
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &x);
		add(i, x);
	}
	for (int i = 1; i <= m; i++)
	{
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			scanf("%d%d", &x, &k);
			add(x, k);
			break;
		case 2:
			scanf("%d%d", &x, &y);
			printf("%d\n", sum(y) - sum(x-1));
			break;
		}
	}
	return 0;
}
