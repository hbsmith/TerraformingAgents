# import TerraformingAgents
using TerraformingAgents
using Agents, Random
using DrWatson: @dict
using Suppressor: @suppress_err

@testset "GalaxyParameters setup" begin

    @test_nowarn TerraformingAgents.GalaxyParameters(10)
    # @test_nowarn TerraformingAgents.GalaxyParameters(nplanets=10) ## I don't know why I can't figure out a way to make this happen, but i'm going to ignore for now
    @test_throws MethodError TerraformingAgents.GalaxyParameters(10.0)

    # # TerraformingAgents.GalaxyParameters(pos=[(1,2)],vel=[(1,2)],planetcompositions=hcat([1]))
    # # ## ^^This doesn't throw, should it? Seems weird to not allow construction with just pos, yet allow...
    # # TerraformingAgents.GalaxyParameters(Random.default_rng(),pos=[(1,2)])
    # # ## Or is it? I guess with the above ^^ you're acknowledging there's randomness in your setup,
    # # ## Whereas if you specify all of pos, vel, compositions, then everything is determined and it 
    # # ## then makes sense you don't need to provide a RNG for setup
    # # ## With that philosophy, then maybe I shouldn't allow something like this after all...
    # # TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0)) 
    # # ## ... because it silently does randomization during parameter assignments

    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141),10)
    @test_throws ArgumentError TerraformingAgents.GalaxyParameters(MersenneTwister(3141),-1)

    pos = [(0.1, 0.1),(0.2, 0.2)]
    vel = [(2.0, 2.0),(2.0, 3.0)]
    planetcompositions = hcat([[0,0,0],[1,0,2]]...)
    compsize = length(planetcompositions[:,1])

    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos)
    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), vel=vel, compsize=compsize, planetcompositions=planetcompositions)
    @test_throws TypeError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=(0,1)) ## Pos is tuple
    @test_throws TypeError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=[(0.1,1)]) ## Pos mixed type
    @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, compsize=compsize, planetcompositions=planetcompositions)
    @test_throws ArgumentError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=[(2.0, 2.0)]) ## Mismatched arg lengths

end

@testset "Initialize planetary systems" begin 

    params = TerraformingAgents.GalaxyParameters(MersenneTwister(3141),10)

    @testset "simple no warn" begin

        @test_nowarn TerraformingAgents.galaxy_model_setup(params)
        
    end

    @testset "pos/vel/composition specific no warn" begin 

        ## For advanced only
        pos = [(0.1, 0.1),(0.2, 0.2)]
        vel = [(2.0, 2.0),(2.0, 3.0)]
        planetcompositions = hcat([[0,0,0],[1,0,2]]...)
        compsize = length(planetcompositions[:,1])

        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos)
        @test_nowarn TerraformingAgents.galaxy_model_setup(params)
            
        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), vel=vel, compsize=compsize, planetcompositions=planetcompositions)
        @test_nowarn TerraformingAgents.galaxy_model_setup(params)

        galaxyparams = TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, compsize=compsize, planetcompositions=planetcompositions)
        @test_nowarn TerraformingAgents.galaxy_model_setup(params)
            
    end

end

@testset "Initialize life" begin

    dt = 1.0
    extent = (1,1) ## Size of space
    interaction_radius = 0.02 
    allowed_diff = 10
    lifespeed = 0.3 ## distance threshold used to decide where to send life from parent planet
    pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)]
    ool = 3

    #################

    galaxyparams = TerraformingAgents.GalaxyParameters(
        MersenneTwister(3141),
        dt=dt,
        extent=extent,
        interaction_radius=interaction_radius,
        allowed_diff=allowed_diff,
        lifespeed=lifespeed,
        pos=pos,
        ool=ool)

    model = TerraformingAgents.galaxy_model_setup(galaxyparams)

    ### Test neighbors exist
    lifeagents = filter(p->isa(p.second,Life),model.agents)
    @test length(lifeagents) == 1
    @test model.agents[5].destination == model.agents[2]
    @test model.agents[5].ancestors == Planet[]
    @test model.agents[2].claimed == true

    ## @test no compatible planets

end

@testset "compatible planets" begin 

    dt = 1.0
    extent = (1,1) ## Size of space
    interaction_radius = 0.02 
    allowed_diff = 3
    lifespeed = 0.3
    pos = [(0.0, 0.0),(0.2, 0.0),(0.2, 0.2),(0.5, 0.5)]
    planetcompositions = hcat([[0,0,0],[1,0,2],[3,3,3],[7,7,7]]...)
    compsize = length(planetcompositions[:,1])
    ool = Int[] ## Length=0 vector to make sure that life doesn't get initialized. A little hacky but this is for testing.

    #################

    galaxyparams = TerraformingAgents.GalaxyParameters(
        MersenneTwister(3141),
        dt=dt,
        extent=extent,
        interaction_radius=interaction_radius,
        allowed_diff=allowed_diff,
        lifespeed=lifespeed,
        pos=pos,
        compsize=compsize,
        planetcompositions=planetcompositions,
        ool=ool)

    model = TerraformingAgents.galaxy_planet_setup(galaxyparams)

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

    ###############################################################################################
    ## Modify allowed_diff and make sure results change
    allowed_diff = 2
    
    galaxyparams = TerraformingAgents.GalaxyParameters(
        MersenneTwister(3141),
        dt=dt,
        extent=extent,
        interaction_radius=interaction_radius,
        allowed_diff=allowed_diff,
        lifespeed=lifespeed,
        pos=pos,
        compsize=compsize,
        planetcompositions=planetcompositions,
        ool=ool)

    model = TerraformingAgents.galaxy_planet_setup(galaxyparams)    

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
    
    agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)  ## if model.dt is 10, and lifespeed = 0.2, then the agent goes .2 per step
    galaxyparams = TerraformingAgents.GalaxyParameters(
        MersenneTwister(3141), 
        dt = 1.0,
        lifespeed = 0.2,
        interaction_radius = 0.02,
        allowed_diff = 3,
        pos = [(.5,.5),(.5,.4),(.5,.3)],
        planetcompositions = hcat([[3,2,1],[8,7,6],[6,3,3]]...),
        compsize = 3,
        ool = 1
        )
    model = galaxy_model_setup(galaxyparams)

    @show model.dt
    
    steps = 0
    n = 2

    # dump(model.agents)

    lifeagents = filter(p->isa(p.second,Life),model.agents)
    @show keys(lifeagents)
    @show keys(model.agents)

    for i in 1:n:20
        step!(model, agent_step!, galaxy_model_step!, n)
        steps+=n
        lifeagents = filter(p->isa(p.second,Life),model.agents)
        @show keys(lifeagents)
        @show keys(model.agents)

        steps == 2 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 4 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 6 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 8 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
        steps == 10 && @test 4 ∉ keys(model.agents) && 5 in keys(model.agents)
        steps == 12 && @test 4 ∉ keys(model.agents) && 5 in keys(model.agents)
    end

end

# # @testset "pos_is_inside_alive_radius" begin
    
# #     agent_step!(agent, model) = move_agent!(agent, model, model.dt)
# #     rng = MersenneTwister(3141)
# #     galaxyparams = GalaxyParameters(
# #         rng,
# #         100,
# #         extent = (100,100),
# #         dt = 10,
# #         allowed_diff = 7,
# #         maxcomp = 16,
# #         compsize = 6)
# #     model = galaxy_model_setup(galaxyparams)

# #     lifedict = filter(kv -> kv.second isa Life, model.agents)
# #     for (id,life) in lifedict
# #         testpos = life.pos

# #         ## Unfinished??


# # end

# @testset "mantel" begin

#     ## Same test used in skbio here:
#     ## https://github.com/biocore/scikit-bio/blob/ecdfc7941d8c21eb2559ff1ab313d6e9348781da/skbio/stats/distance/_mantel.py
#     ## http://scikit-bio.org/docs/0.5.3/generated/generated/skbio.stats.distance.mantel.html
#     rng = MersenneTwister(3141)
#     x = [[0,1,2],[1,0,3],[2,3,0]]
#     y = [[0, 2, 7],[2, 0, 6],[7, 6, 0]]
#     corr_coeff, p_value = TerraformingAgents.MantelTest(hcat(x...),hcat(y...),rng=rng)
#     @test round(corr_coeff, digits=5) == 0.75593
#     @test p_value == 0.666

# end

# @testset "PlanetMantelTest" begin
    
#     agent_step!(agent, model) = move_agent!(agent, model, model.dt)
#     rng = MersenneTwister(3141)
#     galaxyparams = GalaxyParameters(
#         rng,
#         100,
#         extent = (100,100),
#         dt = 10,
#         allowed_diff = 7,
#         maxcomp = 16,
#         compsize = 6)
#     model = galaxy_model_setup(galaxyparams)
#     corr_coeff, p_value = TerraformingAgents.PlanetMantelTest(model)
#     println(corr_coeff)
#     println(p_value)

#     @test_nowarn corr_coeff

# end

# @testset "Propogation of model rng" begin
    
#     ## First model creation
#     agent_step!(agent, model) = move_agent!(agent, model, model.dt)
#     rng = MersenneTwister(3141)
#     galaxyparams = GalaxyParameters(
#         rng,
#         100,
#         extent = (100,100),
#         dt = 10,
#         allowed_diff = 7,
#         maxcomp = 16,
#         compsize = 6)
#     model = galaxy_model_setup(galaxyparams)
#     corr_coeff, p_value = TerraformingAgents.PlanetMantelTest(model)

#     ## Second model creation
#     rng = MersenneTwister(3141)
#     galaxyparams = GalaxyParameters(
#         rng,
#         100,
#         extent = (100,100),
#         dt = 10,
#         allowed_diff = 7,
#         maxcomp = 16,
#         compsize = 6)
#     model = galaxy_model_setup(galaxyparams)
#     corr_coeff2, p_value2 = TerraformingAgents.PlanetMantelTest(model)
#     @test corr_coeff == corr_coeff2
#     @test p_value == p_value2
# end

# # @testset "galaxy model basic no error" begin
    
# #     agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
# #     model = galaxy_model_basic(10, RNG = MersenneTwister(3141))
# #     for i in 1:1:20
# #         step!(model, agent_step!, galaxy_model_step!)
# #     end

# # end

# # @testset "galaxy model basic w/modified planet compositions" begin
    
# #     agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)
# #     model = galaxy_model_basic(10, RNG = MersenneTwister(3141), compositionmaxvalue = 16, compositionsize = 6)
# #     for a in values(model.agents)
# #         @test length(a.composition) == 6
# #         @test maximum(a.composition) <= 16
# #     end
# #     for i in 1:1:20
# #         step!(model, agent_step!, galaxy_model_step!)
# #     end

# # end