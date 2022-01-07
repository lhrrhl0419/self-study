//luogu P5905 【模板】Johnson 全源最短路
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<queue>
#define MAXN 3022
#define MAXM 12022
#define INF (1000000000)
using namespace std;
struct e { int next, from, to, dis; } edge[MAXM];
struct node {
	int p, dis;
};
struct cmp {
	bool operator()(node a, node b) {
		return a.dis > b.dis;
	}
};
priority_queue<node, vector<node>, cmp>pq;
int head[MAXN], dis[MAXN];
int dis2[MAXN];
bool vis[MAXN];
int edgenum = 0;
int n, m, u, v, w;
void addedge(int f, int t, int dis)
{
	edge[++edgenum] = { head[f],f,t,dis };
	head[f] = edgenum;
	return;
}
bool bellman_ford()
{
	for (int i = 1; i <= n+1; i++)
	{
		for (int j = 1; j <= edgenum; j++)
		{
			if (dis[edge[j].to] > dis[edge[j].from] + edge[j].dis)
			{
				dis[edge[j].to] = dis[edge[j].from] + edge[j].dis;
			}
		}
	}	
	for (int j = 1; j <= edgenum; j++)
	{
		if (dis[edge[j].to] > dis[edge[j].from] + edge[j].dis)
		{
			return false;
		}
	}
	return true;
}
void dijkstra(int s)
{
	pq.push({ s,0 });
	dis2[s] = 0;
	int p, t;
	while (pq.empty() == false)
	{
		p = pq.top().p;
		pq.pop();
		if (vis[p])
			continue;
		vis[p] = true;
		for (int i = head[p]; i; i = edge[i].next)
		{
			t = edge[i].to;
			if (vis[t])continue;
			if (dis2[t] > dis2[p] + edge[i].dis)
				dis2[t] = dis2[p] + edge[i].dis, pq.push({ t,dis2[t] });
		}
	}
	return;
}
int main()
{
	scanf("%d%d", &n, &m);
	dis[0] = 0;
	for (int i = 1; i <= n; i++)dis[i] = INF;
	for (int i = 1; i <= m; i++)scanf("%d%d%d", &u, &v, &w), addedge(u, v, w);
	for (int i = 1; i <= n; i++)addedge(0, i, 0);
	if (!bellman_ford())
	{
		printf("-1\n");
		return 0;
	}
	for (int i = 1; i <= m; i++)
		edge[i].dis += dis[edge[i].from] - dis[edge[i].to];
	for (int i = 1; i <= n; i++)
	{
		long long ans = 0;
		for (int j = 1; j <= n; j++)dis2[j] = INF, vis[j]=false;
		dijkstra(i);
		for (int j = 1; j <= n; j++)
		{
			if(dis2[j]!=INF)dis2[j] += dis[j] - dis[i];
			ans += (long long)j * dis2[j];
		}
		printf("%lld\n", ans);
	}
	return 0;
}
