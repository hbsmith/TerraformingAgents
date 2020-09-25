# import TerraformingAgents
using TerraformingAgents
using Agents, Random
using DrWatson: @dict
using Suppressor: @suppress_err


@testset "Check provided args includes non-nothing value" begin
    
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2.0)], 
        :vel => [(1.1, 2.2), (0.1, 2.0)], 
        :planetcompositions => [[[1,2,3],[4,5,6]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## planetcompositions with different lengths
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2.0)], 
        :vel => [(1.1, 2.2), (0.1, 2.0)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## various nothing args
    argdict = Dict(
        :pos => nothing, 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict(:planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    argdict = Dict(
        :pos => [(1, 2), (3.3, 2.0)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict( :pos => [(1, 2), (3.3, 2.0)], :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

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
        :pos => [(1, 2),(3.0,4.1)], 
        :vel => [(1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) == false
        
    @test_throws MethodError TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2),(3.0,4.1)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) 

end

@testset "Initialize planetary systems" begin 
    
    extent = (1,1) ## Size of space
    dt = 1.0 
    psneighbor_radius = 0.2 ## distance threshold used to decide where to send life from parent planet
    interaction_radius = 1e-4 ## how close life and destination planet have to be to interact
    similarity_threshold = 0.5 ## how similar life and destination planet have to be for terraformation

    @testset "Basic" begin

        nplanetspersystem = 1

        @testset "simple no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planetarysystems_basic!(model, 10; @dict(RNG , nplanetspersystem)...)
            
        end

        @testset "negative planets throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
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

    @testset "Advanced" begin 

        ## For advanced only
        pos = [(0.1, 0.1),(0.2, 0.2)]
        vel = [(2.0, 2.0),(2.0, 3.0)]
        planetcompositions = [[[0,0,0]],[[1,0,2]]]

        @testset "pos only, no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planetarysystems_advanced!(model; @dict(RNG , pos)...)
            
        end

        @testset "vel, planetcompositions no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planetarysystems_advanced!(model; @dict(RNG , vel, planetcompositions)...)
            
        end

        @testset "Int pos tuple throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_throws TypeError TerraformingAgents.initialize_planetarysystems_advanced!(model; pos=[(0,1)] ,@dict(RNG )...)
            
        end

        @testset "pos mixed-type tuple throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_throws TypeError TerraformingAgents.initialize_planetarysystems_advanced!(model; pos=[(0.1,1)] ,@dict(RNG )...)
            
        end

        @testset "pos, vel, planetcompositions no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planetarysystems_advanced!(model; @dict(RNG , pos, vel, planetcompositions)...)
            
        end

        @testset "heterogeneous nplanets per ps no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{PlanetarySystem,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    similarity_threshold, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planetarysystems_advanced!(model; planetcompositions = [[[0,0,0],[3,2,9]],[[1,0,2]]], @dict(RNG , pos, vel)...)
            
        end

    end

end

@testset "Initialize psneighbors" begin 

    extent = (1,1) ## Size of space
    @testset "raidus 0.29" begin 
        psneighbor_radius = 0.29 ## distance threshold used to decide where to send life from parent planet
        space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
        model = @suppress_err AgentBasedModel(
            Union{PlanetarySystem,Life}, 
            space2d, 
            properties = @dict(psneighbor_radius))

        RNG = MersenneTwister(3141)
        TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
        TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
        
        @test model.agents[1].neighbors == [2]
        @test model.agents[2].neighbors == [1]
        @test model.agents[3].neighbors == []

    end
    
    @testset "radius 0.3" begin
        psneighbor_radius = 0.3
        space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
        model = @suppress_err AgentBasedModel(
            Union{PlanetarySystem,Life}, 
            space2d, 
            properties = @dict(psneighbor_radius))

        RNG = MersenneTwister(3141)
        TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
        TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
        
        @test model.agents[1].neighbors == [2]
        @test model.agents[2].neighbors == [1]
        @test model.agents[3].neighbors == []

    end

    @testset "radius .43" begin
        psneighbor_radius = 0.43

        space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
        model = @suppress_err AgentBasedModel(
            Union{PlanetarySystem,Life}, 
            space2d, 
            properties = @dict(psneighbor_radius))

        RNG = MersenneTwister(3141)
        TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
        TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
        
        @test model.agents[1].neighbors == [2]
        @test Set(model.agents[2].neighbors) == Set([1,3])
        @test model.agents[3].neighbors == [2]
    end

end

@testset "Initialize nearest neighbor" begin 

    extent = (1,1) ## Size of space
    psneighbor_radius = 0.3 ## distance threshold used to decide where to send life from parent planet
    space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
    model = @suppress_err AgentBasedModel(
        Union{PlanetarySystem,Life}, 
        space2d, 
        properties = @dict(psneighbor_radius))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
    TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
    TerraformingAgents.initialize_nearest_neighbor!(model)

    @test model.agents[1].nearestps == 2
    @test model.agents[2].nearestps == 1 ## I think this should always return the lower index if tied
    @test model.agents[3].nearestps == 2
    @test model.agents[4].nearestps == 3

end

@testset "Initialize life with neighbors" begin

    extent = (1,1) ## Size of space
    lifespeed = 0.2
    psneighbor_radius = 0.3 ## distance threshold used to decide where to send life from parent planet
    space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
    model = @suppress_err AgentBasedModel(
        Union{PlanetarySystem,Life}, 
        space2d, 
        properties = @dict(psneighbor_radius))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
    TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
    TerraformingAgents.initialize_nearest_neighbor!(model)
    TerraformingAgents.initialize_life!(model.agents[3], model, lifespeed)
    ### Test neighbors exist
    lifeagents = filter(p->isa(p.second,Life),model.agents)
    @test length(lifeagents) == 2
    destinations = []
    for i in values(lifeagents)
        push!(destinations, i.destination)
        @test i.parentplanet == 3
    end
    @test Set(destinations) == Set([1,2])

end

@testset "Initialize life without neighbors" begin

    extent = (1,1) ## Size of space
    lifespeed = 0.2
    psneighbor_radius = 0.1 ## distance threshold used to decide where to send life from parent planet
    space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
    model = @suppress_err AgentBasedModel(
        Union{PlanetarySystem,Life}, 
        space2d, 
        properties = @dict(psneighbor_radius))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planetarysystems_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
    TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
    TerraformingAgents.initialize_nearest_neighbor!(model)
    TerraformingAgents.initialize_life!(model.agents[2], model, lifespeed)
    ### Test neighbors exist
    lifeagents = filter(p->isa(p.second,Life),model.agents)
    @test length(lifeagents) == 1
    @test model.agents[5].parentplanet == 2
    @test model.agents[5].destination == 1

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