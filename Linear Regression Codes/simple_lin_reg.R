library(MASS) # R famous 'Boston' dataset
str(Boston)

#help(Boston) detailed info regarding each variable in the dataset

y <- Boston$medv #medv is median home value for each tract

# Matrix of feature variables from Boston
X <- as.matrix(Boston[-ncol(Boston)])

# vector of ones with same length as rows in Boston
int <- rep(1, length(y))

# Add intercept column to X
X <- cbind(int, X)

# %*%  operator denotes matrix multiplication. t()  takes the transpose of a matrix, 
# solve() calculates the inverse of a matrix that is invertible. Use known formula for beta

# Implement closed-form solution
betas <- solve(t(X) %*% X) %*% t(X) %*% y

# Round for easier viewing
betas <- round(betas, 2)
print(betas)

# Linear regression model
lm.mod <- lm(medv ~ ., data=Boston)

# Round for easier viewing
lm.betas <- round(lm.mod$coefficients, 2)

# Create data.frame of results, and compare with results using lm() function
results <- data.frame(our.results=betas, lm.results=lm.betas)

print(results)