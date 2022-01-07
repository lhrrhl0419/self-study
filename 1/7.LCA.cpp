//luogu 【模板】最近公共祖先（LCA）
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<cstring>
#define MAXN 500010
#define MAXLOGN 20
struct e { int next, to; } edge[MAXN << 1];
int head[MAXN];
int edgenum = 0;
int lca[MAXN][MAXLOGN];
int depth[MAXN], dlist[MAXN];
int n, m, s, x, y;
using namespace std;
void addedge(int f, int t)
{
	edge[++edgenum] = { head[f],t };
	head[f] = edgenum;
	edge[++edgenum] = { head[t],f };
	head[t] = edgenum;
	return;
}
void cal_lca(int p)
{
	dlist[depth[p]] = p;
	for (int i = 0; (1 << i) <= depth[p]; i++)
		lca[p][i] = dlist[depth[p] - (1 << i)];
	for (int i = head[p]; i; i = edge[i].next)
	{
		if (depth[p] > 0 && edge[i].to == dlist[depth[p] - 1])continue;
		depth[edge[i].to] = depth[p] + 1;
		cal_lca(edge[i].to);
	}
	return;
}
int sim(int x, int y, int t)
{
	if (t == 0)
	{
		while (x != y)
			x = lca[x][0], y = lca[y][0];
		return x;
	}
	if (depth[x] >= (1 << t) && lca[x][t] != lca[y][t])return sim(lca[x][t], lca[y][t], t - 1);
	else return sim(x, y, t - 1);
}
int main()
{
	scanf("%d%d%d", &n, &m, &s);
	memset(head, 0, sizeof(head));
	for (int i = 1; i < n; i++)scanf("%d%d", &x, &y), addedge(x, y);
	depth[s] = 0;
	cal_lca(s);
	int temp;
	for (int i = 1; i <= m; i++)
	{
		scanf("%d%d", &x, &y);
		if (depth[x] > depth[y])swap(x, y);
		for (int j = MAXLOGN - 1; j >= 0; j--)
		{
			if (depth[y] == depth[x])break;
			if (depth[y] - depth[x] < (1 << j))continue;
			y = lca[y][j];
		}
		if (x == y)
		{
			printf("%d\n", x);
			continue;
		}
		temp = sim(x, y, MAXLOGN);
		//if (temp == 0)printf("!\n");
		printf("%d\n", temp);
	}
	return 0;
}
