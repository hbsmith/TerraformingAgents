# import TerraformingAgents
using TerraformingAgents
using Agents, Random
using DrWatson: @dict
using Suppressor: @suppress_err

@testset "GalaxyParameters setup" begin

    @test_nowarn TerraformingAgents.GalaxyParameters(10)
    # @test_nowarn TerraformingAgents.GalaxyParameters(nplanets=10) ## I don't know why I can't figure out a way to make this happen, but i'm going to ignore for now
    @test_throws MethodError TerraformingAgents.GalaxyParameters(10.0)

    # TerraformingAgents.GalaxyParameters(pos=[(1,2)],vel=[(1,2)],planetcompositions=hcat([1]))
    # ## ^^This doesn't throw, should it? Seems weird to not allow construction with just pos, yet allow...
    # TerraformingAgents.GalaxyParameters(Random.default_rng(),pos=[(1,2)])
    # ## Or is it? I guess with the above ^^ you're acknowledging there's randomness in your setup,
    # ## Whereas if you specify all of pos, vel, compositions, then everything is determined and it 
    # ## then makes sense you don't need to provide a RNG for setup
    # ## With that philosophy, then maybe I shouldn't allow something like this after all...
    # TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0)) 
    # ## ... because it silently does randomization during parameter assignments

    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141),10)
    @test_throws ArgumentError TerraformingAgents.GalaxyParameters(MersenneTwister(3141),-1)

    pos = [(0.1, 0.1),(0.2, 0.2)]
    vel = [(2.0, 2.0),(2.0, 3.0)]
    planetcompositions = hcat([[0,0,0],[1,0,2]]...)

    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos)
    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), vel=vel, planetcompositions=planetcompositions)
    @test_throws TypeError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=(0,1)) ## Pos is tuple
    @test_throws TypeError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=[(0.1,1)]) ## Pos mixed type
    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, planetcompositions=planetcompositions)
    @test_throws ArgumentError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=[(2.0, 2.0)]) ## Mismatched arg lengths

end

@testset "Initialize planetary systems" begin 
    
    extent = (1,1) ## Size of space
    dt = 1.0 
    lifespeed = 0.2 ## distance threshold used to decide where to send life from parent planet
    interaction_radius = 1e-4 ## how close life and destination planet have to be to interact
    allowed_diff = 3 ## how similar life and destination planet have to be for terraformation

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
                Union{Planet,Life}, 
                space2d, 
                properties = @dict(
                    dt, 
                    interaction_radius, 
                    allowed_diff, 
                    lifespeed))

    @testset "simple no warn" begin

        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141),10)
        @test_nowarn TerraformingAgents.initialize_planets!(model, galaxyparams)
        
    end

    @testset "Advanced" begin 

        ## For advanced only
        pos = [(0.1, 0.1),(0.2, 0.2)]
        vel = [(2.0, 2.0),(2.0, 3.0)]
        planetcompositions = hcat([[0,0,0],[1,0,2]]...)

        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos)
        @test_nowarn TerraformingAgents.initialize_planets!(model, galaxyparams)
            
        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), vel=vel, planetcompositions=planetcompositions)
        @test_nowarn TerraformingAgents.initialize_planets!(model, galaxyparams)

        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, planetcompositions=planetcompositions)
        @test_nowarn TerraformingAgents.initialize_planets!(model, galaxyparams)
            
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

    galaxyparams = TerraformingAgents.GalaxyParameters(
        MersenneTwister(3141),
        pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)])
    TerraformingAgents.initialize_planets!(model, galaxyparams)
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
    planetcompositions = hcat([[0,0,0],[1,0,2],[3,3,3],[7,7,7]]...)

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

    galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos = pos, planetcompositions=planetcompositions)
    TerraformingAgents.initialize_planets!(model, galaxyparams)
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

    galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos = pos, planetcompositions=planetcompositions)
    TerraformingAgents.initialize_planets!(model, galaxyparams)
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

@testset "Agent dies at correct planet" begin
    
    agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
    RNG = MersenneTwister(3141)
    galaxyparams = TerraformingAgents.GalaxyParameters(
        RNG, 
        interaction_radius = 0.02,
        allowed_diff = 3,
        pos = [(.5,.5),(.5,.4),(.5,.3)],
        planetcompositions = hcat([[3,2,1],[8,7,6],[6,3,3]]...),
        ool = 1
        )
    model = galaxy_model_setup(RNG, galaxyparams)
    
    steps = 0
    n = 2
    for i in 1:n:20
        step!(model, agent_step!, galaxy_model_step!, n)
        steps+=n
        lifeagents = filter(p->isa(p.second,Life),model.agents)

        steps == 2 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 4 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 6 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 8 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 10 && @test 4 ∉ keys(model.agents) && 5 in keys(model.agents)
        steps == 12 && @test 4 ∉ keys(model.agents) && 5 in keys(model.agents)
    end

end

@testset "pos_is_inside_alive_radius" begin
    
    agent_step!(agent, model) = move_agent!(agent, model, model.dt)
    rng = MersenneTwister(3141)
    galaxyparams = GalaxyParameters(
        rng,
        100,
        extent = (100,100),
        dt = 10,
        allowed_diff = 7,
        maxcomp = 16,
        compsize = 6)
    model = galaxy_model_setup(galaxyparams)

    lifedict = filter(kv -> kv.second isa Life, model.agents)
    for (id,life) in lifedict
        testpos = life.pos


end

# @testset "galaxy model basic no error" begin
    
#     agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
#     model = galaxy_model_basic(10, RNG = MersenneTwister(3141))
#     for i in 1:1:20
#         step!(model, agent_step!, galaxy_model_step!)
#     end

# end

# @testset "galaxy model basic w/modified planet compositions" begin
    
#     agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
#     model = galaxy_model_basic(10, RNG = MersenneTwister(3141), compositionmaxvalue = 16, compositionsize = 6)
#     for a in values(model.agents)
#         @test length(a.composition) == 6
#         @test maximum(a.composition) <= 16
#     end
#     for i in 1:1:20
#         step!(model, agent_step!, galaxy_model_step!)
#     end

# end