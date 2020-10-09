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
    allowed_diff = 3 ## how similar life and destination planet have to be for terraformation

    @testset "Basic" begin

        nplanetspersystem = 1

        @testset "simple no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planets_basic!(10, model; @dict(RNG)...)
            
        end

        @testset "negative planets throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))
            
            RNG = MersenneTwister(3141)
            @test_throws ArgumentError TerraformingAgents.initialize_planets_basic!(-1, model; @dict(RNG)...)

            ## test trying to initialize with 
            ## - negative or 0 planets 
            ## 

        end

    end

    @testset "Advanced" begin 

        ## For advanced only
        pos = [(0.1, 0.1),(0.2, 0.2)]
        vel = [(2.0, 2.0),(2.0, 3.0)]
        planetcompositions = [[0,0,0],[1,0,2]]

        @testset "pos only, no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , pos)...)
            
        end

        @testset "vel, planetcompositions no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , vel, planetcompositions)...)
            
        end

        @testset "Int pos tuple throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_throws TypeError TerraformingAgents.initialize_planets_advanced!(model; pos=[(0,1)] ,@dict(RNG )...)
            
        end

        @testset "pos mixed-type tuple throws" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_throws TypeError TerraformingAgents.initialize_planets_advanced!(model; pos=[(0.1,1)] ,@dict(RNG )...)
            
        end

        @testset "pos, vel, planetcompositions no warn" begin

            space2d = ContinuousSpace(2; periodic = true, extend = extent)
            model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    psneighbor_radius))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , pos, vel, planetcompositions)...)
            
        end

    end

end

@testset "Initialize life with neighbors" begin

    extent = (1,1) ## Size of space
    lifespeed = 0.2
    psneighbor_radius = 0.3 ## distance threshold used to decide where to send life from parent planet
    space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(psneighbor_radius))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planets_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
    TerraformingAgents.spawn_life!(model.agents[3], model, lifespeed)
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

# @testset "approaching_planet" begin 

#     extent = (1,1) ## Size of space
#     lifespeed = 0.2
#     psneighbor_radius = 0.2 ## distance threshold used to decide where to send life from parent planet
#     space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
#     model = @suppress_err AgentBasedModel(
#         Union{Planet,Life}, 
#         space2d, 
#         properties = @dict(psneighbor_radius))

#     RNG = MersenneTwister(3141)
#     TerraformingAgents.initialize_planets_advanced!(model; pos = [(0.1, 0.1),(0.2, 0.2)], RNG = RNG)
#     TerraformingAgents.initialize_psneighbors!(model, psneighbor_radius)  
#     TerraformingAgents.initialize_nearest_neighbor!(model)
    
#     TerraformingAgents.initialize_life!(model.agents[2], model, lifespeed)
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[1]) == true
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[2]) == false
#     model.agents[3].vel = .-model.agents[3].vel
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[1]) == false
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[2]) == false
#     model.agents[3].vel = (-0.2, 0.0)
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[1]) == true
#     @test TerraformingAgents.approaching_planet(model.agents[3],model.agents[2]) == false
    
#     kill_agent!(model.agents[3], model)
#     TerraformingAgents.initialize_life!(model.agents[1], model, lifespeed)
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[2]) == true
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[1]) == false
#     model.agents[4].vel = .-model.agents[4].vel
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[2]) == false
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[1]) == false
#     model.agents[4].vel = (0.0, 0.2)
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[2]) == true
#     @test TerraformingAgents.approaching_planet(model.agents[4],model.agents[1]) == false    

# end

# @testset "Agent dies at correct planet" begin
    
#     agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
#     model = galaxy_model_basic(3, RNG=MersenneTwister(3141), interaction_radius = 0.02)
#     steps = 0
#     for i in 1:2:100
#         step!(model, agent_step!, galaxy_model_step!, 2)
#         steps+=1
#         lifeagents = filter(p->isa(p.second,Life),model.agents)
#         steps == 3 && @test length(lifeagents) == 1
#         steps == 4 && @test length(lifeagents) == 1
#         steps == 5 && @test length(lifeagents) == 0
#     end

#     model = galaxy_model_basic(3, RNG=MersenneTwister(3141), interaction_radius = 0.01)
#     steps = 0
#     for i in 1:2:100
#         step!(model, agent_step!, galaxy_model_step!, 2)
#         steps+=1
#         lifeagents = filter(p->isa(p.second,Life),model.agents)
#         steps == 3 && @test length(lifeagents) == 1
#         steps == 4 && @test length(lifeagents) == 1
#         steps == 5 && @test length(lifeagents) == 0
#     end


# end


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