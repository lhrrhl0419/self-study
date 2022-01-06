//luogu P3372 【模板】线段树 1
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 110000
using namespace std;
int n, m, op, x, y;
long long k;
struct tree { long long sum = 0, lazy = 0; }tr[MAXN << 1];
long long num[MAXN];
void build_tree(int p, int l, int r)
{
	if (l == r)
	{
		tr[p].sum = num[l];
		return;
	}
	int m = (l + r) >> 1;
	build_tree(p << 1, l, m);
	build_tree((p << 1) + 1, m + 1, r);
	tr[p].sum = tr[p << 1].sum + tr[(p << 1) + 1].sum;
	return;
}
void pushdown(int p,int l,int m,int r)
{
	tr[p << 1].lazy += tr[p].lazy;
	tr[(p << 1) + 1].lazy += tr[p].lazy;
	tr[p << 1].sum += tr[p].lazy * ((long long)m - l + 1);
	tr[(p << 1) + 1].sum += tr[p].lazy * ((long long)r - m);
	tr[p].lazy = 0;
	return;
}
void add(int p, int l, int r, int el, int er, long long k)
{
	if (l == r)
	{
		tr[p].sum += k;
		return;
	}
	int m = (l + r) >> 1;
	pushdown(p,l,m,r);
	tr[p].sum += ((long long)er - el + 1) * k;
	if (l == el && r == er)
	{
		tr[p].lazy += k;
		return;
	}
	if (el <= m)add(p << 1, l, m, el, min(m, er), k);
	if (er > m)add((p << 1) + 1, m + 1, r, max(el, m + 1), er, k);
	return;
}
long long sum(int p, int l, int r, int el, int er)
{
	if (el==l && er==r)
	{
		return tr[p].sum;
	}
	int m = (l + r) >> 1;
	pushdown(p,l,m,r);
	long long temp = 0;
	if (el <= m)temp+=sum(p << 1, l, m, el, min(m, er));
	if (er > m)temp+=sum((p << 1) + 1, m + 1, r, max(el, m + 1), er);
	return temp;
}
int main()
{
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)
	{
		scanf("%lld", &num[i]);
	}
	build_tree(1, 1, n);
	for (int i = 1; i <= m; i++)
	{
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			scanf("%d%d%lld", &x, &y, &k);
			add(1, 1, n, x, y, k);
			break;
		case 2:
			scanf("%d%d", &x, &y);
			printf("%lld\n", sum(1, 1, n, x, y));
			break;
		}
	}
	return 0;
}
