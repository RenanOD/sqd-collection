using Main.LDLFactorizations, BenchmarkTools, Test, SparseArrays, LinearAlgebra, MatrixMarket, DelimitedFiles

open("benchmark.log", "w") do logfile
      println(logfile, "Problem & tref & mem_ref & t/tref & mem/mem_ref \\")

problems = [
  "aug2d";
  "aug2dc";
  "aug2dcqp";
  "aug2dqp";
  "aug3d";
  "aug3dc";
  "aug3dcqp";
  "aug3dqp";
  "cvxqp1_s";
  "cvxqp1_m";
  "cvxqp1_l";
  "cvxqp2_s";
  "cvxqp2_m";
  "cvxqp2_l";
  "cvxqp3_s";
  "cvxqp3_m";
  "cvxqp3_l";
  "dual1";
  "dual2";
  "dual3";
  "dual4";
  "dualc1";
  "dualc2";
  "dualc5";
  "dualc8";
  "genhs28";
  "gouldqp2";
  "gouldqp3";
  "hs118";
  "hs21";
  "hs21mod";
  "hs268";
  "hs35";
  "hs35mod";
  "hs51";
  "hs52";
  "hs53";
  "hs76";
  "hues-mod";
  "huestis";
  "ksip";
  "liswet1";
  "liswet10";
  "liswet11";
  "liswet12";
  "liswet2";
  "liswet3";
  "liswet4";
  "liswet5";
  "liswet6";
  "liswet7";
  "liswet8";
  "liswet9";
  "lotschd";
  "mosarqp1";
  "mosarqp2";
  "powell20";
  "primal1";
  "primal2";
  "primal3";
  "primal4";
  "primalc1";
  "primalc2";
  "primalc5";
  "primalc8";
  "qpcblend";
  "qpcboei1";
  "qpcboei2";
  "qpcstair";
  "s268";
  "stcqp1";
  "stcqp2";
  "tame";
  "ubh1";
  "yao";
  "zecevic2"
           ]
  for p in problems

    # read
    A = MatrixMarket.mmread("sqd-collection/$p/3x3/iter_0/K_0.mtx")
    rhs = readdlm("sqd-collection/$p/3x3/iter_0/rhs_0.rhs")[:]
    Aonlyupper = triu(A) # this takes longer than the factorization oO

    LDLref = ldl(A)
    LDLonlyupper = ldl(Aonlyupper, onlyupper = true)

    # tests
    @test norm(A * (LDLref \ rhs) - rhs) < 1e-6
    @test norm(A * (LDLonlyupper \ rhs) - rhs) < 1e-6

    # benchmarks

    benchmark_ref = @benchmark ldl($A)                                         samples=10 evals=1
    benchmark_onlyupper = @benchmark ldl($Aonlyupper, onlyupper = true)        samples=10 evals=1

    tref = round(benchmark_ref.times[1] / 1e9, digits = 6)
    memref = round(benchmark_ref.memory[1] / 1e6, digits = 6)
    trel = round(benchmark_onlyupper.times[1] / benchmark_ref.times[1], digits = 3)
    memrel = round(benchmark_onlyupper.memory[1] / benchmark_ref.memory[1], digits = 3)

    println(logfile, " $p & $(tref) s & $(memref) MiB & $(trel) & $(memrel) \\ ")
  end
end
