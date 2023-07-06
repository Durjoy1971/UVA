/*  -------------------------      Namo Buddhaya        ------------------------   */

#include <bits/stdc++.h>
using namespace std;

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

// template<class T> using oset = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> ordered_set1; // less => Small to Big
typedef tree<int, null_type, greater<int>, rb_tree_tag, tree_order_statistics_node_update> ordered_set2;
// For Pair
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set3;

/* Time Complexity: O(log n)
        - Order_of_key(k): Number of items strictly smaller then k
        => name.order_of_key(100);
        - find_by_order(k): K-th element in a set ( Counting from zero)
        => *name.find_by_order(5);
*/

#define nl "\n"
#define sp " "
#define ll long long int
#define ld long double
#define PI 2 * acos(0.0)
#define tani(a) atan(a) / (PI / 180)
#define sini(a) asin(a) / (PI / 180)
#define cosi(a) acos(a) / (PI / 180)
#define cos(a) cos(a *PI / 180)
#define sin(a) sin(a *PI / 180)
#define tan(a) tan(a *PI / 180)
#define mem(arr, fix) memset(arr, fix, sizeof(arr))
#define eps 1e-6
#define mkp make_pair
#define valid(nx, ny) (nx >= 0 && nx < n) && (ny >= 0 && ny < n)
#define Dpos(n) fixed << setprecision(n)
#define ff first
#define FF first
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

// for contest
typedef tree<pair<ll, int>, null_type, less<pair<ll, int>>, rb_tree_tag, tree_order_statistics_node_update> os01; // Less -> Small To Big
typedef tree<pair<ll, int>, null_type, greater<pair<ll, int>>, rb_tree_tag, tree_order_statistics_node_update> os02;

// Down-Up-Right-Left
int fx[] = {1, -1, 0, 0};
int fy[] = {0, 0, 1, -1};
// Down-Up-Right-Left-Kona-Koni
int fkx[8] = {1, -1, 0, 0, 1, -1, 1, -1};
int fky[8] = {0, 0, 1, -1, 1, -1, -1, 1};
// Pagla Ghora :)
int cx[8] = {2, 2, 1, 1, -1, -1, -2, -2};
int cy[8] = {1, -1, 2, -2, 2, -2, 1, -1};

ll gcd(ll a, ll b)
{
    if (a == 0)
        return b;
    return gcd(b % a, a);
}
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }
ll sum_digit(ll x)
{
    ll sum = 0;
    while (x > 0)
    {
        sum += x % 10;
        x /= 10;
    }
    return sum;
}
ll combination(ll a, ll b)
{
    if ((a == b) || (b == 0))
    {
        return 1;
    }
    if (a < b)
        return 0;
    ll ret = 1;
    for (ll i = 0; i < b; i++)
    {
        ret *= (a - i);
        ret %= mod;
        ret *= bigmod(i + 1, mod - 2, mod);
        ret %= mod;
    }
    return ret;
}
ll factorial(ll n)
{
    ll i, ans = 1;
    for (i = n; i > 1; i--)
    {
        ans *= i;
    }
    return ans;
}
bool palin(string s)
{
    int len = s.size();
    for (int i = 0; i < len; i++)
    {
        if (s[i] != s[len - 1 - i])
        {
            return false;
        }
    }
    return true;
}
ll digit(ll num)
{
    ll n = 0;
    while (num > 0)
    {
        num = num / 10;
        n++;
    }
    return n;
}
bool prime_checker(int num)
{
    if (num == 2)
        return true;
    if (num < 2 || num % 2 == 0)
        return false;
    for (ll i = 2; i * i <= num; i++)
    {
        if (num % i == 0)
            return false;
    }
    return true;
}
ll bigmod(ll b, ll p, ll m)
{
    if (!p)
        return 1;
    ll x = bigmod(b, p / 2, m);
    x = (x * x) % m;
    if (p % 2)
        x = (x * b) % m;
    return x;
}

ll inv_mod(ll a, ll m) {
    return bigmod(a, m-2, m);
}


/*

priority_queue <int> pq; // Boro theke choto
priority_queue<int, vector<int>, greater<int> > q; // choto theke boro

priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>> q;

*/

/*

// Normal Segment Tree With Summation

const ll MAXN = 100002;
ll arr[MAXN];
ll t[4*MAXN];

void build(ll v, ll tl, ll tr)
{ // v = 1, tl = 1 and tr = n // arr[1...n]
    if(tl == tr){
        t[v] = arr[tl];
    }else{
        ll tm = (tl+tr)/2;
        build(v*2,tl,tm);
        build(v*2+1,tm+1,tr);
        t[v] = t[v*2] + t[v*2+1];
    }
}

ll t_sum(ll v, ll tl, ll tr, ll l, ll r)
{ // l,r -> query
    if(l > r) return 0;
    if(l == tl && r == tr) return t[v];
    ll tm = (tl+tr)/2;
    return t_sum(v*2,tl,tm,l,min(r,tm)) + t_sum(v*2+1,tm+1,tr,max(l,tm+1),r);
}

// v-> SegTree r Main Root, tl and tr -> Range Of SegTree
void update(ll v, ll tl, ll tr, ll pos, ll new_val){
    if(tl == tr)
    {
        t[v] = new_val;
    }
    else
    {
        ll tm = (tl+tr)/2;
        if(pos <= tm)
            update(v*2,tl,tm,pos,new_val);
        else update(v*2+1, tm+1, tr, pos, new_val);

        t[v] = t[v*2] + t[v*2+1];
    }
}


*/

/*
// Disjoint Set Union

int p[30];

int Find(int x)
{
    if (p[x] == x)
        return x;

    return p[x] = Find(p[x]);
}

void Union(int a, int b)
{
    p[Find(b)] = Find(a);
}


*/

/*
// Binary Indexed Tree | Fenwick Tree

#define mx 10000
int ar[mx];
int tree[mx];
// Space Complexity O(n)
// Query => O(log n)
// 1 Based Index
int read(int idx){
    int sum = 0;
    while (idx > 0){
        sum += tree[idx];
        idx -= (idx & -idx);
    }
    return sum;
}

void update(int idx, int val, int n){
    while (idx <= n){
        tree[idx] += val;
        idx += (idx & -idx);
    }
}

void print(int *ar, int n) {
    for (int i = 1; i <= n; ++i) {
        cout << ar[i] << " ";
    }
    puts("");
}

// Main Function
 int n; cin >> n;
    for (int i = 1; i <= n; ++i) { cin >> ar[i]; update(i, ar[i], n); }

    cout << "input array:\t";
    print(ar, n);
    cout << "\n";

    cout << "tree array:\t";
    print(tree, n);
    cout << "\n";

*/

/*

// Sparse Table

const int sz = 2e5 + 5;
const int LOG = 18;
int ar[sz];
int m[sz][LOG];
int demo_log[sz];

int query(int left, int right)
{
    int length = right - left + 1;
    int k = demo_log[length];

    return min(m[left][k], m[right-(1<<k)+1][k]);
}

// Main Codes Start Here
void So_Much_Pain()
{
    ll i, j;
    // int i, j;

    int n;
    int q;
    cin >> n;
    cin >> q;

    demo_log[1]=0;
    for(i=2;i<=n;i++)
    {
        demo_log[i] = demo_log[i/2]+1;
    }

    // 1) Read Input
    for (i = 0; i < n; i++)
    {
        cin >> ar[i];
        m[i][0] = ar[i];
    }

    // 2) Preprocessing (O(N*log(N)))
    for (int k = 1; k < LOG; k++)
    {
        for (i = 0; i + (1 << k) - 1 < n; i++)
        {
            m[i][k] = min(m[i][k - 1], m[i + (1 << (k - 1))][k - 1]);
        }
    }

    // 3) Answer Queries
    //cin >> q;
    for (int i = 0; i < q; i++)
    {
        int left, right;
        cin >> left >> right;

        cout << query(left-1, right-1) << nl;
    }
}

*/

/*
    // LCM pair sum with phi

    const int sz = 1e6+10;
int phi[sz];
ll result[sz];
void phi_seive()
{
    int num = sz-5;
    ll i, j;
    for(i=1; i <= num; i++)
    {
        phi[i]=i;
    }
    for(i = 2; i <= num; i++)
    {
        if(phi[i]==i)
        {
            for(int j=i; j<=num;j+=i)
            {
                phi[j] /= i;
                phi[j] *= (i-1);
            }
        }
    }
    //LCM SUM(1 to n-1) => (summation(d*phi(d))
    for(i = 1; i <= sz-5; i++)
    {
        for(j=i; j<= sz-5; j+=i)
        {
            result[j] += phi[i] * i;
        }
    }
    // (sum[n]-1)*n/2
}


*/

/*

    // GCD Pair with Phi [1 to N]

    const int sz = 1e6+7;
int phi[sz];
void phi_seive()
{
    int num = sz-4;
    int i;
    for(i=1; i <= num; i++)
    {
        phi[i]=i;
    }
    for(i = 2; i <= num; i++)
    {
        if(phi[i]==i)
        {
            for(int j=i; j<=num;j+=i)
            {
                phi[j] /= i;
                phi[j] *= (i-1);
            }
        }
    }
}
ll result [sz];
void gcdpair()
{
    ll i, j;
    for(i = 1; i < sz-5; i++)
    {
        for(j = 2; i*j < sz-5; j++)
        {
            result[i*j] += i*phi[j];
        }
    }
    for(i = 2; i < sz-5; i++)
    {
        result[i] += result[i-1];
    }
}

*/

/*

// Euler Segmented Sieve

vector <bool> prime(mx,true);
vector <int> p_num;
ll phi[sz];
ll phi2[sz];
void kipta_sieve(ll a, ll b)
{
    ll i, j;
    prime[0] = prime[1]=false;
    p_num.push_back(2);
    for(i = 3; i <= mx; i=i+2)
    {
        if(prime[i])
        {
            for(j = i*i; j <= mx; j += i+i)
            {
                prime[j] = false;
            }
        }
    }
    for(i = 3; i <= mx; i=i+2)
    {
        if(prime[i]==true)
        {
            p_num.push_back(i);
        }
    }

    for(i = 0; i <= (ll)1e5; i++)
    {
        phi[i] = phi2[i] = a+i;
    }

    for(i = 0; i < (ll)p_num.size(); i++)
    {
        ll num;
        if(p_num[i]<a) num = (ceil(a*1.00/p_num[i]))*(ll)p_num[i];
        else num = p_num[i];
        for(j = num; j <= b; j+=p_num[i])
        {
            phi[j-a] = phi[j-a]/p_num[i];
            phi[j-a] = phi[j-a]*(p_num[i]-1);
            while(phi2[j-a]%p_num[i]==0)
            {
                phi2[j-a] /= p_num[i];
            }
        }
    }

    for(i = a; i <= b; i++)
    {
        if(phi2[i-a] > 1)
        {
            phi[i-a] /= phi2[i-a];
            phi[i-a] *= phi2[i-a]-1;
        }
    }

    for(i = a; i <= b; i++)
    {
        cout << phi[i-a] << nl;
    }

}


*/

/*

// seive prime checker

ll mx = 1LL << 16;
vector <bool> prime(mx,true);
vector <int> p_num;
void kipta_sieve()
{
    ll i, j;
    prime[0] = prime[1]=false;
    p_num.push_back(2);
    for(i = 3; i < mx; i = i+2)
    {
        if(prime[i])
        {
            for(j = i*i; j < mx; j += i+i)
            {
                prime[j] = false;
            }
        }
    }
    for(i = 3; i <= mx; i=i+2)
    {
        if(prime[i]==true)
        {
            p_num.push_back(i);
        }
    }
}
void p_checker(ll num)
{
    for(ll i = 0; p_num[i]*p_num[i] <= num && i < (int)p_num.size(); i++)
    {
        if(num%p_num[i]==0)
        {
            i = -1;
            num--;
        }
    }
    cout << num << nl;
}

*/

/*

// Find out the first digit of N to the power (n)

ll n;

while (cin >> n)
{
    double ans = n * log10(n);
    n = ans;
    cout << (ll)pow(10, ans - n) << nl;
}

*/

/*
    // npr

    int Fac[1000001];
ll mod = 1e9+7;
ll nPr(ll n, ll r)
{
    if(r > n) return 0;

    ll res = Fac[n];
    res = (res * bigmod(Fac[n-r],mod-2,mod))%mod;

    return res;
}
ll nCr(ll n, ll r)
{
    if(r > n) return 0;

    ll res = Fac[n];
    res = (res * bigmod(Fac[r],mod-2, mod))%mod;
    res = (res * bigmod(Fac[n-r],mod-2,mod))%mod;

    return res;
}
// Main Function

int n, r;
    cin >> n >> r;
    Fac[0] = Fac[1] = 1;

    for(int i = 2; i <= 1000000; i++)
    {
        Fac[i] = (Fac[i-1] * 1LL * i)%mod;
    }
    cout << nPr(n,r) << nl;
        cout << nCr(n,r) << nl;
*/
/*
    // Modular Inverse
    a to the power inverse one = bigmod(a,m-2,m);
    where m is prime and (a,m) is co-prime.
*/

/*
    // Euler's Totient Function

    //                              (1)                         //
Novice Approach(O(n* log(n))
int Phi(int num)
{
    int cnt = 0;
    for(int i = 1; i <= num; i++)
    {
        if(__gcd(i,num) == 1)
        {
            cnt++;
        }
    }
    return cnt;
}
// ---------------------------------------------------------------------------- //
//                              (2)                         //
int phi(int n)     //Time ComPlexity(O(sqrt(n)))
{
    int res = n;

    for(int i = 2; i*i<=n; i++)
    {
        if(n % i == 0)
        {
            res /= i;
            res *= (i-1);
            while(n%i==0)
            {
                n /= i;
            }
        }
    }
    if(n > 1)
    {
        res /= n;
        res = res*(n-1);
    }

    return res;
}
//------------------------------------------------------------------//

//                              (3)                         //
int phi[100005];
void phi_seive()
{
    int num = 100005;
    int i;
    for(i=1; i <= num; i++)
    {
        phi[i]=i;
    }
    for(i = 2; i <= num; i++)
    {
        if(phi[i]==i)
        {
            for(int j=i; j<=num;j+=i)
            {
                phi[j] /= i;
                phi[j] *= (i-1);
            }
        }
    }
}
//------------------------------------------------------------------//
*/
/*
set<int>S;
set<int>::iterator it;

for (it=S.begin();it!= S.end();it++)
    {
        cout<<*it<<endl;
    }

*/
/*
// Combination (nCr)
long long nCr(ll n, ll r)
{
    long long p, k;
    p = k = 1;
    //C(n,r) == C(n,n-r)
    if(n-r < r) r = n - r;
    if(r != 0)
    {
        while(r)
        {
            p *= n;
            k *= r;
            long long m = __gcd(p,k);
            p /= m;
            k /= m;
            n--;
            r--;
        }
    }
    else p = 1;

    return p;
}


*/

/*

    // Simple BFS (Sir)

//Function
vector<int> level;
vector<int> edges[100005];
int n, m;
void bfs(int s)
{
    queue <int> q;
    q.push(s);
    level[s]=0;
    while(!q.empty())
    {
        int u = q.front();
        q.pop();
        for(auto x: edges[u])
        {
            if(level[x] == -1){
                q.push(x);
                level[x] = level[u]+1;
            }
        }
    }
}
//Main Function
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin >> t;
    while(t--)
    {
        int i, j;
        cin >> n >> m;
        for(i = 0; i < n; i++) edges[i].clear();
        for(i = 0; i < m; i++)
        {
            int u, v;
            cin >> u >> v;
            edges[u].push_back(v);
            edges[v].push_back(u);
        }
        level = vector<int>(n+1,-1);
        int cnt = 0;
        for(i = 0; i < n; i++)
        {
            if(level[i]==-1)
            {
                bfs(i);
                cnt++;
            }
        }
        cout << cnt << nl;
    }


    return 0;
}

*/

/*

// Simple BFS

int n, m, c;
vector<int> arr[10001];
bool visited[10001];
int dis[10001];
void BFS(int src)//src = source
{
    queue <int> q;
    q.push(src);
    visited[src]=true;
    dis[src]=0;
    while(!q.empty())
    {
        int curr = q.front();
        q.pop();
        for(int child : arr[curr])
        {
            if(!visited[child])
            {
                q.push(child);
                dis[child] = dis[curr]+1;
                visited[child] = 1;
            }
        }
    }
}


*/

/*

// DFS Circle Detection

#include <bits/stdc++.h>
#include <windows.h>
using namespace std;

#define nl "\n"
#define sp " "
#define PI 2 * acos(0)
#define mem(arr, value) memset(arr, value, sizeof(arr))
#define eps 1e-6
#define ll long long
#define mp make_pair
//Function
int i, j;
int n, m, c;
vector< pair<int,int> > edges[100001];
int visited[100001];
int par[100001];
bool cy;//gray = 1; white = 0; black = 2;
void dfs(int s)
{
    cout << s << sp;
    visited[s]=1;
    for(auto x : edges[s])
    {
        int v = x.first;
        if(visited[v]==1)
        {
            cy = 1;
        }
        if(visited[v]==0)
        {
            par[v]= s;
            dfs(v);
        }
    }
    visited[s]=2;
}

//Main Function
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cin >> n >> m;
    for (i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        edges[u].push_back( make_pair(v,c));
        edges[v].push_back( make_pair(u,c));
    }

    dfs(1);
    cout << nl;
    if(cy) cout << "Cycle detected";
    return 0;
}

// memset(array name, 0, sizeof(array));

*/

/*

// Basic DFS

//Function
int i, j;
int n, m, c;
vector< pair<int,int> > edges[100001];
bool visited[100001];
int par[100001];
void dfs(int s)
{
    cout << s << sp;
    visited[s]=1;
    for(auto x : edges[s])
    {
        int v = x.first;
        if(!visited[v])
        {
            par[v]= s;
            dfs(v);
        }
    }
}

//Main Function
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cin >> n >> m;
    for (i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v >> c;
        edges[u].push_back( make_pair(v,c));
        edges[v].push_back( make_pair(u,c));
    }

    dfs(1);
    cout << nl;
    for(i = 1; i <=n; i++)
    {
        cout << i << sp << par[i] << nl;
    }

    return 0;
}

// memset(array name, 0, sizeof(array));


*/

/*

// Extended GCD


ll egcd(ll a, ll b, ll &x, ll &y)
{
    if(a == 0)
    {
        x = 0; y = 1;
        return b;
    }
    ll x1, y1;
    ll d = egcd(b%a, a, x1, y1);

    x = y1 - (b/a)*x1;
    y = x1;

    //cout <<  nl << x << sp << y << nl;
    return d;
}

*/

/*

// Kipta Seive


int mx = 1e8;
vector <bool> prime(mx,true);
void kipta_sieve()
{
    ll i, j;
    for(i = 3; i < mx; i = i+2)
    {
        if(prime[i])
        {
            for(j = i*i; j < mx; j += i+i)
            {
                prime[j] = false;
            }
        }
    }
}



*/

/*

// Factorization

 ll n;
    cin >> n;
    for(i = 2; i * i <= n; i++)
    {
        if(prime(i) && n % i == 0)
        {
            int cnt = 0;
            while(n % i == 0)
            {
                cnt++;
                n = n / i;
            }
            cout <<"("<< i << "^" << cnt << ")";
            if(n != 1) cout << "*";
        }
    }
    if(n > 1) cout <<"("<< n << "^" << 1 << ")" << nl;




*/

/*

// bitwise exhaustive search

int i, j;
    int n = 3;
    for(i = 0; i < (1 << (n-1)); i++)
    {
        for(j = 0; j <= n; j++)
        {
            cout << (i >> j & 1) << nl;
        }
        cout << nl;
    }
//
when n = 3;
0 0 0 0
1 0 0 0
0 1 0 0
1 1 0 0
//

*/

/*
    // bit masking


    int n, k;cin >> n >> k;
    vector <int> dj(k);for(auto &x : dj) cin >> x;int ans = 0;
    //bitmasking
    for (int mask = 1; mask < (1 << k); mask++)//mask can be 0 also.
    {
        int cnt = 0;
        int div = 1;
        for(int i = 0; i < k; i++)//Main Part.
        {
            if(mask & (1 << i)){
                div = lcm(div, dj[i]);
                cnt++;
            }
        }
        //After com. other's work
        if(cnt % 2 == 1) ans += n / div;
        else ans -= n / div;
    }

    cout << ans << nl;


*/

/*

    // Normal Binary With Upper Bound Then Lower Bound

    Trick to Remember
// Lower Bound -> First pos to insert x ... x <= array r element
//Upper Bound -> Last pos to insert x....  array r element <= x
// Ekta same value koibar ache ber korar Upay : UpperBound - LowerBound
// 1 2 2 3 4 4 5


Upper bound : a[l] <= x and a[r] > x;
Lower bound : a[l] < x and a[r] >= x

ll Lower_bound(vector <int> &arr, ll element)
{
    ll i, j, mid;
    ll hi, lo;
    lo = -1;
    hi = arr.size();
    while(hi-lo > 1)
    {
        mid = lo + (hi-lo)/2;
        //l < x && hi >= x
        if(arr[mid] >= element) hi = mid;
        else lo = mid;
    }
    return hi;
}

ll Upper_bound(vector <int> &arr, ll element)
{
    ll i, j, mid;
    ll hi, lo;
    lo = -1;
    hi = arr.size();
    while(hi-lo > 1)
    {
        mid = lo + (hi-lo)/2;
        //l < x && hi > x
        if(arr[mid] > element) hi = mid;
        else lo = mid;
    }
    return hi;
}


int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (auto &x : a)
    {
        cin >> x;
    }
    for (int i = 0; i < k; i++)
    {
        int x;
        cin >> x;
        bool ok = 0;
        int l = -1, r = n;
        while (l < r - 1)
        {
            int m = (l + r) / 2;
            if (a[m] > x)
            {
                r = m;
            }
            else l = m;
        }
        int djupper = r;
        //cout << upper_bound(a.begin(), a.end(), x)-a.begin() << sp;
        //cout << r << nl;
        //if(r > 0 and )
        l = -1, r = n;
        while (l < r - 1)
        {
            int m = (l + r) / 2;
            if (a[m] >= x)
            {
                r = m;
            }
            else l = m;
      }
        int djlower = r;
        //cout << djlower << sp << djupper << nl;
        if(djlower == djupper) cout << djlower+1 << nl;
        else cout << djlower+1 << nl;
    }


*/

/*

    // Normal Binary

    //    to avoid    overflow : l + (r-l)/2
    int n, q;
    cin >> n >> q;
    vector <int> arr(n);
    for(i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end());
    while(q--)
    {
        int f_num;
        cin >> f_num;
        int lo = 0, hi = n-1;
        bool flag = false;
        while(hi-lo>=0)
        {
            int mid = (lo+hi)/2;
            if(arr[mid]==f_num)
            {
                cout << "found" << nl;
                flag = true;
                break;
            }
            else if(arr[mid] < f_num)
            {
                lo = mid+1;
            }
            else
            {
                hi = mid-1;
            }

        }
        if(!flag) cout << "not found" << nl;
    }


*/

/*

    // Seive

    int prime[300000], nprime;
int mark[1000002];
void seive(int n)
{
    int i, j, limit = sqrt(n*1.00)+2;
    mark[1] = 1; // 0 means prime.
    for(i = 4; i <= n; i=i+2){mark[i] = 1;}
    prime[nprime++]=2;
    for(i = 3; i <= n; i=i+2){if(!mark[i]){prime[nprime++] = i;if(i <= limit){for(j = i * i; j <= n; j += i*2){ mark[j]=1;}}}}
}

*/

/*

    // Sort With Structure

#include <bits/stdc++.h>

using namespace std;

#define nl "\n"
#define sp " "
#define PI 2 * acos(0)

//Function
struct cf
{
    int num1, num2;
};

bool comp(cf n1, cf n2)
{
    if(n1.num1 > n2.num1) return false;
    else if(n1.num1 == n2.num1)
    {
        if(n1.num2 < n2.num2) return false;
        else return true;
    }
    else return true;
}

//Main Function
int main()
{
    //Start
    int t, n;

    scanf("%d", &t);

    while(t--)
    {
        scanf("%d", &n);
        int i, j, x, y;
        vector<cf>vec(n);
        for(i = 0; i < n; i++)
        {
            scanf("%d %d", &vec[i].num1, &vec[i].num2);
        }
        sort(vec.begin(), vec.end(), comp);
        for(i = 0; i < n; i++)
        {
            printf("%d %d\n", vec[i].num1, vec[i].num2);
        }
    }

    return 0;
}

*/

/*

int n, m;
vector<pair<int, int>> edges[105];
ll dis[105];

void dijkstra(int begin, int end)
{
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> qu;

    qu.push({1, dis[1]});

    while (!qu.empty())
    {
        auto x = qu.top();
        qu.pop();
         if(x.ss != dis[x.ff])
        {
            continue;
        }
        // x.ff => source
        // x.ss => distance
        for (auto node : edges[x.ff])
        {
            // node.ff => new source
            // node.ss => cost
            if (dis[node.ff] > dis[x.ff] + node.ss)
            {
                dis[node.ff] = dis[x.ff] + node.ss;
                qu.push(mkp(node.ff,dis[node.ff]));
            }
        }
    }
}
*/

/*

// Used for breaking words
    stringstream s(str);

    // To store individual words
    string word;

    int count = 0;
    while (s >> word)
        count++;
    return count

*/

/*

// Normal Segment Tree With Minimum

const ll MAXN = 100002;
ll arr[MAXN];
ll t[4*MAXN];

void build(ll v, ll tl, ll tr)
{ // v = 1, tl = 1 and tr = n // arr[1...n]
    if(tl == tr){
        t[v] = arr[tl];
    }else{
        ll tm = (tl+tr)/2;
        build(v*2,tl,tm);
        build(v*2+1,tm+1,tr);
        t[v] = min(t[v*2],t[v*2+1]);
    }
}

ll t_minimum(ll v, ll tl, ll tr, ll l, ll r)
{ // l,r -> query
    if(l > r) return 1e18;
    if(l == tl && r == tr) return t[v];
    ll tm = (tl+tr)/2;
    return min(t_minimum(v*2,tl,tm,l,min(r,tm)),t_minimum(v*2+1,tm+1,tr,max(l,tm+1),r));
}

// v-> SegTree r Main Root, tl and tr -> Range Of SegTree
void update(ll v, ll tl, ll tr, ll pos, ll new_val){
    if(tl == tr)
    {
        t[v] = new_val;
    }
    else
    {
        ll tm = (tl+tr)/2;
        if(pos <= tm)
            update(v*2,tl,tm,pos,new_val);
        else update(v*2+1, tm+1, tr, pos, new_val);

        t[v] = min(t[v*2],t[v*2+1]);
    }
}

*/

/*

// Pair Segment Tree With Minimum and Minimum Counting

const ll MAXN = 100002;
ll arr[MAXN];
pair<ll, ll> t[4 * MAXN];

pair<ll, ll> combine(pair<ll, ll> ff, pair<ll, ll> ss)
{
    if (ff.first < ss.first)
    {
        return ff;
    }
    if (ff.first > ss.first)
    {
        return ss;
    }
    return make_pair(ff.first, ff.second + ss.second);
}

void build(ll v, ll tl, ll tr)
{ // v = 1, tl = 1 and tr = n // arr[1...n]
    if (tl == tr)
    {
        t[v] = make_pair(arr[tl], 1LL);
    }
    else
    {
        ll tm = (tl + tr) / 2;
        build(v * 2, tl, tm);
        build(v * 2 + 1, tm + 1, tr);
        t[v] = combine(t[v * 2], t[v * 2 + 1]);
    }
}

pair<ll, ll> get_min(ll v, ll tl, ll tr, ll l, ll r)
{ // l,r -> query
    if (l > r)
        return make_pair(1e18, 0);
    if (l == tl && r == tr)
        return t[v];
    ll tm = (tl + tr) / 2;
    return combine(get_min(v * 2, tl, tm, l, min(r, tm)), get_min(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
}

// v-> SegTree r Main Root, tl and tr -> Range Of SegTree
void update(ll v, ll tl, ll tr, ll pos, ll new_val)
{
    if (tl == tr)
    {
        t[v] = make_pair(new_val, 1LL);
    }
    else
    {
        ll tm = (tl + tr) / 2;
        if (pos <= tm)
            update(v * 2, tl, tm, pos, new_val);
        else
            update(v * 2 + 1, tm + 1, tr, pos, new_val);

        t[v] = combine(t[v * 2], t[v * 2 + 1]);
    }
}


*/

/*

// Segment Tree With Lazy Propagation

ll arr[200005], st[1000005], lazy[1000005];
void build(ll si, ll ss, ll se)
{
    if (ss == se)
    {
        st[si] = arr[ss];
        return;
    }
    ll mid = (ss + se) / 2;
    build(2 * si, ss, mid);
    build(2 * si + 1, mid + 1, se);
    st[si] = st[2 * si] + st[2 * si + 1];
}
void update(ll si, ll ss, ll se, ll qs, ll qe, ll val)
{
    if (lazy[si] != 0)
    {
        ll dx = lazy[si];
        lazy[si] = 0;
        st[si] += (dx * (se - ss + 1));
        if (ss != se)
        {
            lazy[2 * si] += dx;
            lazy[2 * si + 1] += dx;
        }
    }
    if (se < qs || qe < ss)
        return;
    else if (qs <= ss && qe >= se)
    {
        ll dx = val;
        st[si] += (dx * (se - ss + 1));
        if (ss != se)
        {
            lazy[2 * si] += dx;
            lazy[2 * si + 1] += dx;
        }
        return;
    }
    ll mid = (ss + se) / 2;
    update(2 * si, ss, mid, qs, qe, val);
    update(2 * si+1, mid+1, se, qs, qe, val);

    st[si]=st[2*si]+st[2*si+1];
}
ll query(ll si,ll ss,ll se,ll qs,ll qe)
{
    if (lazy[si] != 0)
    {
        ll dx = lazy[si];
        lazy[si] = 0;
        st[si] += (dx * (se - ss + 1));
        if (ss != se)
        {
            lazy[2 * si] += dx;
            lazy[2 * si + 1] += dx;
        }
    }
    if (se < qs || qe < ss)
        return 0;
    else if (qs <= ss && qe >= se)
    {
        return st[si];
    }
    ll mid = (ss + se) / 2;
    return query(2 * si, ss, mid, qs, qe)+query(2 * si+1, mid+1, se, qs, qe);
}

input:

int n, q;
cin >> n >> q;
for (ll i = 1; i <= n; i++)
{
    cin >> arr[i];
}
build(1,1,n);
update(1,1,n,l,r,val);
cout << query(1,1,n,k,k) << endl;

*/

/*

Ternary Search




*/

// Main Codes Start Here
void Solve()
{
    ll i, j;

    ordered_set1 boro_to_choto, choto_to_boro;

    int n;
    cin >> n;
    vi ar;
    for (i = 0; i < n; i++)
    {
        vin(ar);
        boro_to_choto.insert(num * -1);
    }

    /* Time Complexity: O(log n)
        - Order_of_key(k): Number of items strictly smaller then k
        => name.order_of_key(100);
        - find_by_order(k): K-th element in a set ( Counting from zero)
        => *name.find_by_order(5);
    */

    ll ans = 0;

    for (i = n - 1; i >= 0; i--)
    {
        int num = ar[i];
        int left_side_option = boro_to_choto.order_of_key(num * -1);
        boro_to_choto.erase(num * -1);
        choto_to_boro.insert(num);
        int right_side_option = choto_to_boro.order_of_key(num);

        // cout << left_side_option << sp << right_side_option << nl;

        ans += left_side_option * 1LL * right_side_option;
    }

    cout << ans << nl;
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
        //  cout << nl;
        Solve();
    }

    return 0;
}
