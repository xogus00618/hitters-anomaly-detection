library(ISLR)
library(isotree)

set.seed(123)
data(Hitters)
hitters_raw <- na.omit(Hitters)
data_clean <- hitters_raw[!rownames(hitters_raw) == "-Pete Rose",]
data_numeric <- data_clean[, sapply(data_clean, is.numeric)]
hitters <- data_numeric[, colnames(data_numeric) != "Salary"]
col_means <- colMeans(hitters)
col_means[c("CAtBat", "CHits")] <- c(8000, 1000)
synth <- rbind(hitters, col_means)

sample_size <- c(30, 50, 100)
n_trees_list <- c(100, 500, 1000)
track_every <- c(10, 20)

result_PCA_sample <- list()

for (s in sample_size) {
  for (n_trees in n_trees_list) {
    
    # 전체 scores 행렬 저장 (트리 x 관측값)
    score_mat <- matrix(0, nrow = n_trees, ncol = nrow(synth))
      
    for (n in 1:n_trees) {
      idx <- sample(1:nrow(synth), s)
      spl <- synth[idx,]
      pca <- prcomp(spl)
      ax <- predict(pca, newdata = synth)
      iso <- isolation.forest(ax, sample_size = s, ntrees = 1, seed = 123)
      score_mat[n,] <- predict(iso, ax)
    }
    
    # 누적 평균 점수
    cum_mean <- apply(score_mat, 2, cumsum) / (1:n_trees)
    
    # 분산 (관측값별)
    score_var <- apply(score_mat, 2, var)
    var_263 <- score_var[263]
    
    # 10, 20개 단위 순위 추적
    rank_track <- list()
    for (te in track_every) {
      steps <- seq(te, n_trees, by = te)
      ranks_at_steps <- sapply(steps, function(k) {
        mean_k <- colMeans(score_mat[1:k, , drop = FALSE])
        ranked <- order(mean_k, decreasing = TRUE)
        which(ranked == 263)
      })
      rank_track[[paste0("every_", te)]] <- data.frame(
        step = steps,
        rank = ranks_at_steps
      )
    }
    
    # 최종 순위
    mean_score <- colMeans(score_mat)
    ranked <- order(mean_score, decreasing = TRUE)
    rank_263 <- which(ranked == 263)
    
    key <- paste0("ss_", s, "_nt_", n_trees)
    result_PCA_sample[[key]] <- list(
      rank = rank_263,
      score = mean_score[263],
      var_263 = var_263,
      rank_track = rank_track
    )
    
    cat(key, "| rank:", rank_263, "| var:", round(var_263, 6), "\n")
  }
}
print(result_PCA_sample[["ss_30_nt_1000"]]$rank_track)
print(result_PCA_sample[["ss_50_nt_1000"]]$rank_track)
print(result_PCA_sample[["ss_100_nt_1000"]]$rank_track)

for (key in names(result_PCA_sample)) {
  cat(key, "| var_263:", result_PCA_sample[[key]]$var_263, "\n")
}
