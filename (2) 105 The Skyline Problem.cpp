/*  Namo Buddhaya */

#include <bits/stdc++.h>

using namespace std;

#define nl "\n"
#define sp " "
#define ll long long int
#define PI 2 * acos(0.0)
#define mem(arr, fix) memset(arr, fix, sizeof(arr))
#define eps 1e-6
#define mkp make_pair
#define valid(nx, ny) (nx >= 0 && nx < n) && (ny >= 0 && ny < n)
#define Dpos(n) fixed << setprecision(n)
#define ff first
#define ss second
#define vi vector<int>
#define vl vector<long long int>
#define mod ((ll)1e9 + 7)
#define pii pair<int, int>

// Global Variable and Array

// Function
bool comp(pair<pii, int> a, pair<pii, int> b)
{
    if (a.first.first < b.first.first)
    {
        return true;
    }
    else if (a.first.first > b.first.first)
    {
        return false;
    }
    else // 0 means start And 1 means finish
    {    // 2 tay equal hoile
        if (a.second == b.second && a.second == 0)
        {
            if (a.first.second >= b.first.second)
                return true;
            else
                return false;
        }
        else if (a.second == b.second && a.second == 1)
        {
            if (a.first.second < b.first.second)
                return true;
            else
                return false;
        }
        else
        {
            if (a.first.second >= b.first.second)
                return true;
            else
                return false;
        }
    }
}

// Main Codes Start Here
void Pagla()
{
    ll i, j;
    // int i, j;

    int arr[10005] = {0};
    int a, b, h;

    vector<int> maxi;

    vector<pair<pii, int>> ar;

    while (cin >> a >> h >> b)
    {
        ar.push_back(mkp(mkp(a, h), 0));
        ar.push_back(mkp(mkp(b, h), 1));
    }

    sort(ar.begin(), ar.end(), comp);

    /*for(auto x: ar)
    {
        cout << x.ff.ff << sp <<x.ff.ss << nl;
    }*/

    priority_queue<int> pq;

    pq.push(0);
    int mx = 0;
    for (auto x : ar)
    {
        priority_queue<int> cpy;
        if (x.second == 0) // start
        {
            pq.push(x.ff.ss);
            if (pq.top() > mx)
            {
                mx = pq.top();
                cout << x.ff.ff << sp << mx << sp;
            }
        }
        else // end
        {
            bool flag = true;

            while (!pq.empty())
            {
                int num = pq.top();
                pq.pop();
                if (flag)
                {
                    if (num == x.ff.ss)
                    {
                        flag = false;
                    }
                    else
                    {
                        cpy.push(num);
                    }
                }
                else
                    cpy.push(num);
            }
            pq = cpy;
            if(mx != pq.top())
            {
                if(x.ff.ff == ar[(int)ar.size()-1].first.first && pq.top()==0)
                {
                    cout << x.ff.ff << sp << pq.top() << nl;
                }
                else cout << x.ff.ff << sp << pq.top() << sp;
                
                mx = pq.top(); 
            }
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

    while (t--)
    {
        Pagla();
    }

    return 0;
}