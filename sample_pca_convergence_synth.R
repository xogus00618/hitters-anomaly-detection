library(isotree)
set.seed(123)

n <- 10000
x1 <- rnorm(n, mean = 10, sd = 2)
x2 <- x1 + rnorm(n, mean = 0, sd = 0.2)
x3 <- x1 + rnorm(n, mean = 0, sd = 0.2)
x4 <- x1 + rnorm(n, mean = 0, sd = 0.2)
x5 <- x1 + rnorm(n, mean = 0, sd = 0.2)
x_indep <- matrix(rnorm(n * 95, mean = 10, sd = 2), nrow = n, ncol = 95)
colnames(x_indep) <- paste0("x", 6:100)
df <- data.frame(x1, x2, x3, x4, x5, x_indep)

outlier_row <- rep(10, 100)
outlier_row[1:2] <- c(7, 13)
df_with_outlier <- rbind(df, outlier_row)

sample_size <- c(30, 50, 100)
n_trees_list <- c(100, 500, 1000)
track_every <- c(10, 20)

result_PCA_sample <- list()

for (s in sample_size) {
  for (n_trees in n_trees_list) {
    
    score_mat <- matrix(0, nrow = n_trees, ncol = nrow(df_with_outlier))
    
    for (n in 1:n_trees) {
      idx <- sample(1:nrow(df_with_outlier), s)
      spl <- df_with_outlier[idx,]
      pca <- prcomp(spl, scale = TRUE)
      ax <- predict(pca, newdata = df_with_outlier)
      iso <- isolation.forest(ax, sample_size = s, ntrees = 1, seed = 123)
      score_mat[n,] <- predict(iso, ax)
    }
    
    # 분산
    score_var <- apply(score_mat, 2, var)
    var_10001 <- score_var[10001]
    
    # 순위 추적
    rank_track <- list()
    for (te in track_every) {
      steps <- seq(te, n_trees, by = te)
      ranks_at_steps <- sapply(steps, function(k) {
        mean_k <- colMeans(score_mat[1:k, , drop = FALSE])
        ranked <- order(mean_k, decreasing = TRUE)
        which(ranked == 10001)
      })
      rank_track[[paste0("every_", te)]] <- data.frame(
        step = steps,
        rank = ranks_at_steps
      )
    }
    
    # 최종 순위
    mean_score <- colMeans(score_mat)
    ranked <- order(mean_score, decreasing = TRUE)
    rank_10001 <- which(ranked == 10001)
    
    key <- paste0("ss_", s, "_nt_", n_trees)
    result_PCA_sample[[key]] <- list(
      rank = rank_10001,
      score = mean_score[10001],
      var_10001 = var_10001,
      rank_track = rank_track
    )
    
    cat(key, "| rank:", rank_10001, "| var:", round(var_10001, 6), "\n")
  }
}
print(result_PCA_sample[["ss_30_nt_1000"]]$rank_track)
print(result_PCA_sample[["ss_50_nt_1000"]]$rank_track)
print(result_PCA_sample[["ss_100_nt_1000"]]$rank_track)

for (key in names(result_PCA_sample)) {
  cat(key, "| var_10001:", result_PCA_sample[[key]]$var_10001, "\n")
}
