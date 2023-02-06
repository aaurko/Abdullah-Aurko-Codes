# create matrix with 3 rows and 3 columns
Matrix = matrix(1:9, nrow = 3)

# print the matrix
print(Matrix)

# create another matrix
M2 = Matrix

# Loops for Matrix Transpose
for (i in 1:nrow(M2))
{
  # iterate over each row
  for (j in 1:ncol(M2))
  {
    # iterate over each column
    # assign the correspondent elements
    # from row to column and column to row.
    M2[i, j] <- Matrix[j, i]
  }
}

# print the transposed matrix
print(M2)

