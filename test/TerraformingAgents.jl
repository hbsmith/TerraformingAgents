# import TerraformingAgents
using TerraformingAgents
using Agents, Random
using DrWatson: @dict

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

@testset "Initialize planetary systems" begin 

    @testset "Basic no warn" begin

        extent = (1,1) ## Size of space
        dt = 1.0 
        psneighbor_radius = 0.2 ## distance threshold used to decide where to send life from parent planet
        interaction_radius = 1e-4 ## how close life and destination planet have to be to interact
        similarity_threshold = 0.5 ## how similar life and destination planet have to be for terraformation
        nplanetspersystem = 1

        space2d = ContinuousSpace(2; periodic = true, extend = extent)
        model = AgentBasedModel(
            Union{TerraformingAgents.PlanetarySystem,TerraformingAgents.Life}, 
            space2d, 
            properties = @dict(
                dt, 
                interaction_radius, 
                similarity_threshold, 
                psneighbor_radius))

        RNG = MersenneTwister(3141)
        @test_nowarn TerraformingAgents.initialize_planetarysystems_basic!(model, 10; @dict(RNG , nplanetspersystem)...)
        
    end

    @testset "Basic negative planets" begin

        extent = (1,1) ## Size of space
        dt = 1.0 
        psneighbor_radius = 0.2 ## distance threshold used to decide where to send life from parent planet
        interaction_radius = 1e-4 ## how close life and destination planet have to be to interact
        similarity_threshold = 0.5 ## how similar life and destination planet have to be for terraformation
        nplanetspersystem = 1

        space2d = ContinuousSpace(2; periodic = true, extend = extent)
        model = AgentBasedModel(
            Union{TerraformingAgents.PlanetarySystem,TerraformingAgents.Life}, 
            space2d, 
            properties = @dict(
                dt, 
                interaction_radius, 
                similarity_threshold, 
                psneighbor_radius))
        
        RNG = MersenneTwister(3141)
        @test_throws ArgumentError TerraformingAgents.initialize_planetarysystems_basic!(model, -1; @dict(RNG , nplanetspersystem)...)

        ## test trying to initialize with 
        ## - negative or 0 planets 
        ## 

    end

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