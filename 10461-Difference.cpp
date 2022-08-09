/*  -------------------------      Namo Buddhaya        ------------------------   */

/*
    Give The Ones You Love Wings To Fly, Roots To Come Back, And Reasons To Stay.
                    -------------       -------------
                     Love Is The Absence Of Judgment.
                    -------------       -------------
    It's Only Pain Which Lead's Us To Achieve Some Thing Special.
*/

#include <bits/stdc++.h>
using namespace std;

#define debug(i) cout << "Debug " << i << nl
#define nl "\n"
#define sp " "
#define ll long long int
#define ld long double
#define PI 2 * acos(0.0)
#define mem(arr, fix) memset(arr, fix, sizeof(arr))
#define eps 1e-6
#define mkp make_pair
#define valid(nx, ny) (nx >= 1 && nx <= n) && (ny >= 1 && ny <= m)
#define Dpos(n) fixed << setprecision(n)
#define ff first
#define ss second
#define vi vector<int>
#define vl vector<long long int>
#define mod ((ll)1e9 + 7)
#define pii pair<int, int>
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define allrev(v) v.rbegin(), v.rend()
#define vin(name) \
    ll num;       \
    cin >> num;   \
    name.push_back(num);

// Global Variable && Functions
ll v, e;
vector<int> edges[505];
ll coin[505];
bool visit[505];
ll sum;

void dfs(int node)
{
    // cout << node << nl;
    visit[node] = true;
    sum += coin[node];

    for (auto x : edges[node])
    {
        // cout << x << sp;
        if (visit[x] == false)
        {
            dfs(x);
        }
    }
    // cout << nl;
}

int non_dfs(int node, int cnt)
{
    visit[node] = true;

    for (auto x : edges[node])
    {
        if (visit[x] == false)
        {
            non_dfs(x, ++cnt);
        }
    }
    return cnt;
}

// Main Codes Start Here
void Solve()
{
    ll i, j;
    int tt = 1;
    bool flag = false;
    while (cin >> v >> e)
    {
        //cout << "Durjoy";
        if (v == 0 && e == 0)
        {
            return;
        }
        else if(flag) cout << nl;
        flag = true;
        cout << "Case #" << tt++ << ":" << nl;
        for (i = 1; i <= v; i++)
        {
            cin >> coin[i];
        }

        for (i = 1; i <= e; i++)
        {
            edges[i].clear();
            visit[i] = 0;
        }

        for (i = 1; i <= e; i++)
        {
            int u, v;
            cin >> u >> v;
            // cout << u << sp << v << nl;
            edges[u].push_back(v);
        }

        int query;
        cin >> query;

        while (query--)
        {
            mem(visit, false);
            int node;
            cin >> node;

            ll mn = 0;
            sum = 0;
            dfs(node);
            mn = sum;

            int sizing[v + 5] = {0};
            for (i = 1; i <= v; i++)
            {
                if (i == node)
                    continue;
                mem(visit, 0);
                bool flag = true;
                for(auto x : edges[i])
                {
                    if(x == node) flag = false;
                }
                if(flag) sizing[i] = non_dfs(i, 0);
            }

            ll mx = 0;

            for (i = 1; i <= v; i++)
            {
                //cout << "DJ : " << i << sp << edges[i].size() << sp << sizing[i] << nl;
                if (i == node)
                {
                    //cout << "T : " << i << nl;
                    mx += coin[i];
                }
                else if (edges[i].size() == sizing[i])
                {
                    //cout << "T : " << i << nl;
                    mx += coin[i];
                }
            }

            cout << max(0LL,mx-mn) << nl;
        }
    }
}

// Main Function
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    //:]
    int t = 1;

    // cin >> t;

    for (int i = 1; i <= t; i++)
    {
        // cout << "Case " << i << ": ";
        Solve();
    }

    return 0;
}