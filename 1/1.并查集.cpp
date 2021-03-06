//luogu P3367 【模板】并查集
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 11000
using namespace std;
int n, m;
int fa[MAXN];
int find_fa(int x)
{
	if (fa[x] == x)return x;
	fa[x] = find_fa(fa[x]); 
	return temp;
}
int main()
{
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)fa[i] = i;
	int z, x, y;
	for (int i = 1; i <= m; i++)
	{
		scanf("%d%d%d", &z, &x, &y);
		if (z == 1)
			fa[find_fa(x)] = find_fa(y);
		else
		{
			if (find_fa(x) == find_fa(y))printf("Y\n");
			else printf("N\n");
		}
	}
	return 0;
}
