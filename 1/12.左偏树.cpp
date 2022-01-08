//luogu P3377 【模板】左偏树（可并堆）
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<algorithm>
#define MAXN 100010
using namespace std;
int n, m, op, x, y, temp;
int fa[MAXN], lson[MAXN], rson[MAXN], num[MAXN], dist[MAXN];
bool exist[MAXN];
int find_fa(int p)
{
	if (fa[p] == p)return p;
	fa[p] = find_fa(fa[p]);
	return fa[p];
}
int merge(int x, int y)
{
	if (x == 0 || y == 0)
		return x + y;
	if (num[x] > num[y] || (num[x] == num[y] && x > y))swap(x, y);
	rson[x] = merge(rson[x], y);
	if (dist[rson[x]] > dist[lson[x]])swap(lson[x], rson[x]);
	dist[x] = dist[rson[x]] + 1;
	return x;
}
int main()
{
	scanf("%d%d", &n, &m);
	dist[0] = -1;
	for (int i = 1; i <= n; i++)scanf("%d", &num[i]), fa[i] = i, lson[i] = rson[i] = 0, dist[i] = 0, exist[i] = true;
	int fax, fay, res;
	while (m--)
	{
		scanf("%d", &op);
		if (op == 1)
		{
			scanf("%d%d", &x, &y);
			fax = find_fa(x), fay = find_fa(y);
			if (!exist[x] || !exist[y] || fax == fay)continue;
			res = merge(fax, fay);
			fa[fax] = fa[fay] = res;
		}
		else
		{
			scanf("%d", &x);
			if (!exist[x])
			{
				printf("-1\n");
				continue;
			}
			fax = find_fa(x);
			exist[fax] = false;
			printf("%d\n", num[fax]);
			res = merge(lson[fax], rson[fax]);
			fa[fax] = fa[lson[fax]] = fa[rson[fax]] = res;
		}
	}
	return 0;
}
