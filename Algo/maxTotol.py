
# C++ program for Dynamic
# Programming implementation (Top-Down) of
# Max sum problem in a triangle
N = 3

#  Function for finding maximum sum


def maxPathSum(tri, i, j, row, col, dp):
    if (j == col):
        return 0

    if (i == row-1):
        return tri[i][j]

    if (dp[i][j] != -1):
        return -1

    dp[i][j] = tri[i][j] + max(maxPathSum(tri, i+1, j, row, col, dp),
                               maxPathSum(tri, i+1, j+1, row, col, dp))
    return dp[i][j]


# Driver program to test above functions
tri = [[2, 0, 0, 0],
       [3, 4, 0, 0],
       [6, 5, 7, 0],
       [4, 1, 8, 3]]

dp = [[-1 for i in range(N)]for j in range(N)]
print(maxPathSum(tri, 0, 0, N, N, dp))
