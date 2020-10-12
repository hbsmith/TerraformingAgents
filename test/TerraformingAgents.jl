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
    lifespeed = 0.2 ## distance threshold used to decide where to send life from parent planet
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
                    lifespeed))

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
                    lifespeed))
            
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
                    lifespeed))

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
                    lifespeed))

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
                    lifespeed))

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
                    lifespeed))

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
                    lifespeed))

            RNG = MersenneTwister(3141)
            @test_nowarn TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , pos, vel, planetcompositions)...)
            
        end

    end

end

@testset "Initialize life" begin

    dt = 1.0
    extent = (1,1) ## Size of space
    interaction_radius = 0.02 
    allowed_diff = 10
    lifespeed = 0.3 ## distance threshold used to decide where to send life from parent planet
    space2d = ContinuousSpace(2; periodic = true, extend = extent, metric = :euclidean)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(
            dt,
            interaction_radius,
            allowed_diff,
            lifespeed))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planets_advanced!(model; pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)], RNG = RNG)
    TerraformingAgents.spawnlife!(model.agents[3], model)
    ### Test neighbors exist
    lifeagents = filter(p->isa(p.second,Life),model.agents)
    @test length(lifeagents) == 1
    @test model.agents[5].destination == model.agents[2]
    @test model.agents[5].ancestors == Planet[]
    @test model.agents[2].claimed == true

    ## @test no compatible planets

end

@testset "compatible planets; nearest compatible planets" begin 

    dt = 1.0
    extent = (1,1) ## Size of space
    interaction_radius = 0.02 
    lifespeed = 0.3
    pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)]
    planetcompositions = [[0,0,0],[1,0,2],[3,3,3],[7,7,7]]

    #################
    allowed_diff = 3
    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            allowed_diff, 
            lifespeed))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , pos, planetcompositions)...)
    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[1], model)
    @test Set(candidateplanets) == Set([model.agents[2], model.agents[3]])
    @test TerraformingAgents.nearestcompatibleplanet(model.agents[1], candidateplanets) == model.agents[2]

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[2], model)
    @test Set(candidateplanets) == Set([model.agents[1], model.agents[3]])
    @test TerraformingAgents.nearestcompatibleplanet(model.agents[2], candidateplanets) == model.agents[1]

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[3], model)
    @test Set(candidateplanets) == Set([model.agents[1], model.agents[2]])
    @test TerraformingAgents.nearestcompatibleplanet(model.agents[3], candidateplanets) == model.agents[2]

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[4], model)
    @test Set(candidateplanets) == Set()
    @test_throws ArgumentError TerraformingAgents.nearestcompatibleplanet(model.agents[4], candidateplanets)

    #################
    allowed_diff = 2
    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            allowed_diff, 
            lifespeed))

    RNG = MersenneTwister(3141)
    TerraformingAgents.initialize_planets_advanced!(model; @dict(RNG , pos, planetcompositions)...)
    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[1], model)
    @test Set(candidateplanets) == Set([model.agents[2]])
    @test TerraformingAgents.nearestcompatibleplanet(model.agents[1], candidateplanets) == model.agents[2]

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[2], model)
    @test Set(candidateplanets) == Set([model.agents[1]])
    @test TerraformingAgents.nearestcompatibleplanet(model.agents[2], candidateplanets) == model.agents[1]

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[3], model)
    @test Set(candidateplanets) == Set()
    @test_throws ArgumentError  TerraformingAgents.nearestcompatibleplanet(model.agents[3], candidateplanets)

    candidateplanets = TerraformingAgents.compatibleplanets(model.agents[4], model)
    @test Set(candidateplanets) == Set()
    @test_throws ArgumentError TerraformingAgents.nearestcompatibleplanet(model.agents[4], candidateplanets)


end 

@testset "mix compositions" begin
    
    @test TerraformingAgents.mixcompositions([0,0,0],[1,0,2]) == [0,0,1]
    @test TerraformingAgents.mixcompositions([1,1,1,9],[8,8,9,2]) == [4,4,5,6]
    @test TerraformingAgents.mixcompositions([8],[8]) == [8]
    @test TerraformingAgents.mixcompositions([8,9],[4,2]) == [6,6]

end

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
#                     :lifespeed => .45,
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