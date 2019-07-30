using Main.LDLFactorizations, BenchmarkTools, Test, SparseArrays, LinearAlgebra,
      MatrixMarket, DelimitedFiles, DataFrames, SolverBenchmark

# load problems and create columns
include("problems.jl")
np = length(problems)
tref = zeros(np)
memref = zeros(np)
trel = zeros(np)
memrel = zeros(np)

for (i, p) in enumerate(problems)
  # read
  A = MatrixMarket.mmread("sqd-collection/$p/3x3/iter_0/K_0.mtx")
  rhs = readdlm("sqd-collection/$p/3x3/iter_0/rhs_0.rhs")[:]
  Aonlyupper = triu(A)

  LDLref = ldl(A)
  LDLonlyupper = ldl(Aonlyupper, onlyupper = true)

  # test
  @test norm(A * (LDLref \ rhs) - rhs) < 1e-6
  @test norm(A * (LDLonlyupper \ rhs) - rhs) < 1e-6

  # benchmark
  benchmark_ref = @benchmark ldl($A)                                         samples=10 evals=1
  benchmark_onlyupper = @benchmark ldl($Aonlyupper, onlyupper = true)        samples=10 evals=1

  tref[i] = round(benchmark_ref.times[1] / 1e9, digits = 6)
  memref[i] = round(benchmark_ref.memory[1] / 1e6, digits = 4)
  trel[i] = round(benchmark_onlyupper.times[1] / benchmark_ref.times[1], digits = 3)
  memrel[i] = round(benchmark_onlyupper.memory[1] / benchmark_ref.memory[1], digits = 3)
end

benchmarktable = DataFrame(problem = problems, tref_s = tref, memref_MB = memref, trel = trel, memrel = memrel)

open("benchmark.log", "w") do logfile
  println(logfile, "Avg trel: $(sum(trel)/np)")
  println(logfile, "Avg memrel: $(sum(memrel)/np)")
  println(logfile, "So using only the upper triangular is on average $(round((1 - sum(trel)/np)*100, digits = 1)) % faster while using $(round((1 - sum(memrel)/np)*100, digits = 1)) % less memory.")

  markdown_table(logfile, benchmarktable)
end
