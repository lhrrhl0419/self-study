//luogu P3383 【模板】线性筛素数
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 100000010
using namespace std;
int n, q, k;
int prime[MAXN >> 3];
int num = 0;
bool is_prime[MAXN];
int main()
{
	scanf("%d%d", &n, &q);
	for (int i = 2; i <= n; i++)is_prime[i] = true;
	for (int i = 2; i <= n; i++)
	{
		if (is_prime[i])
			prime[++num] = i;
		for (int j = 1; j <= num && i * prime[j] <= n; j++)
		{
			is_prime[i * prime[j]] = false;
			if (i % prime[j] == 0)break;
		}
	}
	for (int i = 1; i <= q; i++)
	{
		scanf("%d", &k);
		printf("%d\n", prime[k]);
	}
	return 0;
}
