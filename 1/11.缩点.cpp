//luogu P3387 【模板】缩点
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<queue>
#define MAXN 22000
#define MAXM 220010
using namespace std;
struct e { int next, from, to; } edge[2][MAXM];
int head[2][MAXN], fa[MAXN], weight[MAXN], depth[MAXN], dlist[MAXN], topo[MAXN], dp[MAXN];
queue<int>q;
bool vis[MAXN];
int edgenum[2];
int n, m, u, v;
void addedge(int f, int t, int type)
{
	edge[type][++edgenum[type]] = { head[type][f],f,t };
	head[type][f] = edgenum[type];
	return;
}
int find_fa(int p)
{
	if (fa[p] == p)return p;
	fa[p] = find_fa(fa[p]);
	return fa[p];
}
void tarjan(int p)
{
	int t;
	vis[p] = true;
	dlist[depth[p]] = p;
	for (int i = head[0][p]; i; i = edge[0][i].next)
	{
		t = find_fa(edge[0][i].to);
		if (vis[t] && dlist[depth[t]] == t && depth[t] <= depth[p])
		{
			for (int j = depth[p]; j >= 0; j--)
			{
				j = depth[find_fa(dlist[j])];
				if (dlist[j] == t)break;
				fa[find_fa(dlist[j])] = t;
			}
		}
		else if (!vis[t])
		{
			depth[t] = depth[p] + 1;
			tarjan(t);
		}
	}
	dlist[depth[p]] = 0;
	return;
}
int main()
{
	edgenum[0] = edgenum[1] = 0;
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)scanf("%d", &weight[i]), fa[i] = i, vis[i] = false, topo[i] = 0, depth[i] = -1, head[0][i] = head[1][i] = 0;
	for (int i = 1; i <= m; i++)scanf("%d%d", &u, &v), addedge(u, v, 0);
	for (int i = 1; i <= n; i++)
	{
		if (vis[i] == true)continue;
		depth[i] = 0;
		tarjan(i);
	}
	for (int i = 1; i <= m; i++)
	{
		if (find_fa(edge[0][i].from) == find_fa(edge[0][i].to))continue;
		topo[find_fa(edge[0][i].to)]++;
		addedge(find_fa(edge[0][i].from), find_fa(edge[0][i].to), 1);
	}
	for (int i = 1; i <= n; i++)
		if (find_fa(i) != i)
			weight[find_fa(i)] += weight[i];
	while (q.empty() == false)q.pop();
	for (int i = 1; i <= n; i++)
		if (topo[i] == 0 && find_fa(i) == i)
			q.push(i), dp[i] = weight[i];
	int p, t, ans = 0;
	while (q.empty() == false)
	{
		p = q.front();
		q.pop();
		ans = max(ans, dp[p]);
		for (int i = head[1][p]; i; i = edge[1][i].next)
		{
			t = edge[1][i].to;
			dp[t] = max(dp[t], dp[p] + weight[t]);
			topo[t]--;
			if (!topo[t])q.push(t);
		}
	}
	printf("%d\n", ans);
	return 0;
}
