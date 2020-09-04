# import TerraformingAgents
using TerraformingAgents

@testset "Check provided args includes non-nothing value" begin
    
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => [(1.1, 2.2), (0.1, 2)], 
        :planetcompositions => [[[1,2,3],[4,5,6]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## planetcompositions with different lengths
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => [(1.1, 2.2), (0.1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## various nothing args
    argdict = Dict(
        :pos => nothing, 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict(:planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict( :pos => [(1, 2), (3.3, 2)], :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test_throws ArgumentError TerraformingAgents.providedargs(Dict(
        :pos => nothing, 
        :vel => nothing, 
        :planetcompositions => nothing))

end

@testset "Check if args have identical lengths" begin
    
    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2), (3,4)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]]))

    ## Length mismatches                                                                   
    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) == false

    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2),(3,4.1)], 
        :vel => [(1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) == false
        
    @test_throws MethodError TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2),(3,4.1)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) 

end

# @testset "Initialization" begin

#     args = Dict(:nplanetarysystems => 10)

#     modelparams = Dict(:RNG => MersenneTwister(1236),
#                     :psneighbor_radius => .45,
#                     :dt => 0.1)

#     model = galaxy_model(
#         args[:nplanetarysystems]
#         ;modelparams...)

#     ## give pos, use default vel, comp
#     ## give vel, use defualt pos, comp
#     ## give pos and vel, use default comp 
#     ## give pos and comp, use default vel 
#     ## give 


# end