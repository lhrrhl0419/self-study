//luogu P3385 【模板】负环
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<queue>
#define MAXN 2022
#define MAXM 6022
#define INF (1<<29)
using namespace std;
struct e { int next, to, dis; } edge[MAXM];
queue<int>q;
int head[MAXN], cnt[MAXN], dis[MAXN];
bool vis[MAXN];
int edgenum;
int n, m, u, v, w;
void addedge(int f, int t, int dis)
{
	edge[++edgenum] = { head[f],t,dis };
	head[f] = edgenum;
	if (dis < 0)return;
	edge[++edgenum] = { head[t],f,dis };
	head[t] = edgenum;
	return;
}
bool spfa()
{
	int p, t;
	while (q.empty() == false)
	{
		p = q.front();
		q.pop();
		vis[p] = false;
		for (int i = head[p]; i; i = edge[i].next)
		{
			t = edge[i].to;
			if (dis[t] > dis[p] + edge[i].dis)
			{
				dis[t] = dis[p] + edge[i].dis;
				cnt[t] = cnt[p] + 1;
				if (cnt[t] > n)return true;
				if (!vis[t])q.push(t);
			}
		}
	}
	return false;
}
int main()
{
	int t;
	scanf("%d", &t);
	while (t--)
	{
		scanf("%d%d", &n, &m);
		edgenum = 0;
		for (int i = 1; i <= n; i++)head[i] = 0, cnt[i] = 0, dis[i] = INF, vis[i] = false;
		while (q.empty() == false)q.pop();
		for (int i = 1; i <= m; i++)scanf("%d%d%d", &u, &v, &w), addedge(u, v, w);
		dis[1] = 0, vis[1] = true, q.push(1);
		if (spfa())printf("YES\n");
		else printf("NO\n");
	}
	return 0;
}
