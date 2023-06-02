function TestGalaxyParametersSetup()
    @testset "GalaxyParameters Setup" begin

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
        @test_throws MethodError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=(0,1)) ## Pos is tuple
        @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=[(0.1,1)]) ## Pos mixed type OK
        @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, compsize=compsize, planetcompositions=planetcompositions)
        @test_throws ArgumentError TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=[(2.0, 2.0)]) ## Mismatched arg lengths

        planetcompositions = hcat([[0.,0.,0.],[1.,0.,2.]]...)
        @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), vel=vel, compsize=compsize, planetcompositions=planetcompositions)
        @test_nowarn TerraformingAgents.GalaxyParameters(MersenneTwister(3141), pos=pos, vel=vel, compsize=compsize, planetcompositions=planetcompositions)

    end
end

function TestInitializePlanetarySystems()
    @testset "Initialize Planetary Systems" begin 

        params = TerraformingAgents.GalaxyParameters(MersenneTwister(3141),10)

        @testset "simple no warn" begin

            @test_logs min_level=Logging.Warn TerraformingAgents.galaxy_model_setup(params)
            
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
end

function TestInitializeLife()
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
end

function TestCompatiblePlanets()
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
end

function TestMixCompositions()
    @testset "mix compositions" begin
        
        @test TerraformingAgents.average_compositions([0,0,0],[1,0,2]) == [0,0,1]
        @test TerraformingAgents.average_compositions([1,1,1,9],[8,8,9,2]) == [4,4,5,6]
        @test TerraformingAgents.average_compositions([8],[8]) == [8]
        @test TerraformingAgents.average_compositions([8,9],[4,2]) == [6,6]

    end
end

function TestCrossoverOnePoint()
    @testset "crossover_one_point" begin
        
        @test TerraformingAgents.crossover_one_point([0,0,0],[1,1,1],1) == ([0,1,1],[1,0,0])
        @test TerraformingAgents.crossover_one_point([0,0,0],[1,1,1],2) == ([0,0,1],[1,1,0])
        @test_throws BoundsError TerraformingAgents.crossover_one_point([0,0,0],[1,1,1],4)
        ## first randint between 1:10 is 3, second randint between 0:1 is 1
        @test TerraformingAgents.crossover_one_point(zeros(10), ones(10), MersenneTwister(3143), mutation_rate=0) == [1,1,1,0,0,0,0,0,0,0]
        
        ## positions to mutate below should be [0,0,0,0,0,0,1,1,0,0]
        crossover_strand = TerraformingAgents.crossover_one_point(zeros(10), ones(10), MersenneTwister(3))
        @test findall(x->x∉[0,1], crossover_strand) == [7,8]
    end
end

function TestMutation()
    @testset "positions_to_mutate" begin
        random_strand =  [0.8116984049958615,
            0.9884323655013432,
            0.8076220876500786,
            0.9700908450487538,
            0.14006111319509862,
            0.5094438024440222,
            0.05869740597593154,
            0.004257960600515309,
            0.9746379934512355,
            0.5572251384524507]

        @test TerraformingAgents.positions_to_mutate(random_strand) == [0,0,0,0,0,0,1,1,0,0]
        
        ## should return the same positions_to_mutate as above
        mutated_strand = TerraformingAgents.mutate_strand(ones(10),1,MersenneTwister(3))
        @test findall(x->x!=1,mutated_strand) == [7,8]
    end
end

function TestAgentDiesAtCorrectPlanet()
    @testset "Agent dies at correct planet" begin
        
        # agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)  ## if model.dt is 10, and lifespeed = 0.2, then the agent goes .2 per step
        galaxyparams = TerraformingAgents.GalaxyParameters(
            MersenneTwister(3141), 
            dt = 0.1,
            lifespeed = 0.2,
            interaction_radius = 0.02,
            allowed_diff = 3,
            pos = [(.5,.5),(.5,.4),(.5,.3)],
            planetcompositions = hcat([[3,2,1],[8,7,6],[6,3,3]]...),
            compsize = 3,
            ool = 1
            )
        
        model = galaxy_model_setup(galaxyparams)    
        steps = 0
        n = 2

        for i in 1:n:20
            step!(model, galaxy_agent_step!, galaxy_model_step!, n)
            steps+=n
            lifeagents = filter(p->isa(p.second,Life),model.agents)

            steps == 2 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
            steps == 4 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
            steps == 6 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
            steps == 8 && @test 4 in keys(model.agents) && 5 ∉ keys(model.agents)
            steps == 10 && @test 4 ∉ keys(model.agents) 
            steps == 12 && @test 4 ∉ keys(model.agents) 
        end

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

function TestCenterPositions()
    @testset "center positions" begin 
        @test TerraformingAgents.center_position((1,),(9,),3) == (4.0,)
        @test TerraformingAgents.center_position((1,1),(9,9),3) == (4.0, 4.0)
        @test TerraformingAgents.center_position((1,1,1),(9,9,9),3) == (4.0, 4.0, 4.0)
    end
end

function TestMantel()
    @testset "mantel" begin

        ## Same test used in skbio here:
        ## https://github.com/biocore/scikit-bio/blob/ecdfc7941d8c21eb2559ff1ab313d6e9348781da/skbio/stats/distance/_mantel.py
        ## http://scikit-bio.org/docs/0.5.3/generated/generated/skbio.stats.distance.mantel.html
        rng = MersenneTwister(3141)
        x = [[0,1,2],[1,0,3],[2,3,0]]
        y = [[0, 2, 7],[2, 0, 6],[7, 6, 0]]
        corr_coeff, p_value = TerraformingAgents.MantelTest(hcat(x...),hcat(y...),rng=rng)
        @test round(corr_coeff, digits=5) == 0.75593
        @test p_value == 0.666

    end

    @testset "mantel vegan" begin

        ## Same test used in skbio here:
        ## https://github.com/biocore/scikit-bio/blob/master/skbio/stats/distance/tests/test_mantel.py#L247
        rng = MersenneTwister(3141)
        veg_dm_vegan = readdlm(joinpath(@__DIR__,"data","mantel_veg_dm_vegan.txt"))
        env_dm_vegan = readdlm(joinpath(@__DIR__,"data","mantel_env_dm_vegan.txt"))
        corr_coeff, p_value = TerraformingAgents.MantelTest(veg_dm_vegan,env_dm_vegan,rng=rng,alternative=:greater)
        @show corr_coeff
        @show p_value
        @show round(corr_coeff, digits=7)
        @test round(corr_coeff, digits=7) == 0.3047454
        @test p_value == 0.001
    end
end

function TestPlanetMantelTest()
    @testset "PlanetMantelTest" begin
        
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
        corr_coeff, p_value = TerraformingAgents.PlanetMantelTest(model)

        @test_nowarn corr_coeff

    end
end

function TestPropogationOfModelRNG()
    @testset "Propogation of model rng" begin
        
        ## First model creation
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
        corr_coeff, p_value = TerraformingAgents.PlanetMantelTest(model)

        ## Second model creation
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
        corr_coeff2, p_value2 = TerraformingAgents.PlanetMantelTest(model)
        @test corr_coeff == corr_coeff2
        @test p_value == p_value2
    end
end

function TestRunningModelNoErrors()
    @testset "modify compmix_func and compmix_kwargs" begin 

        galaxyparams = GalaxyParameters(
            MersenneTwister(3141),
            100,
            extent = (100,100),
            dt = 10,
            allowed_diff = .5,
            maxcomp = 1,
            compsize = 10,
            compmix_func=crossover_one_point,
            compmix_kwargs=Dict(:mutation_rate=>0))
        model = galaxy_model_setup(galaxyparams)    
        n = 100
        adata =  [:pos,
            :vel,
            :composition, # property of Planet and Life
            :initialcomposition, # todo rename as initial_composition
            :alive,
            :claimed,
            :parentcompositions,
            :destination_distance]
        
        @test_logs min_level=Logging.Warn df_agent, df_model = run!(model, 
            galaxy_agent_step!, 
            galaxy_model_step!, 
            n,
            adata=adata, 
            showprogress=true)
    end
end

function TestCandidatePlanetFuncs()
    @testset "compatible planets" begin 

        dt = 1.0
        extent = (1,1) ## Size of space
        interaction_radius = 0.02 
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
            lifespeed=lifespeed,
            pos=pos,
            compsize=compsize,
            planetcompositions=planetcompositions,
            ool=ool)

        model = TerraformingAgents.galaxy_planet_setup(galaxyparams)
        planet = model.agents[1]

        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test Set(map(p -> p.id, values(candidateplanets))) == Set([2,3,4])

        model.agents[2].claimed = true
        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test Set(map(p -> p.id, values(candidateplanets))) == Set([3,4])

        model.agents[3].alive = true
        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test Set(map(p -> p.id, values(candidateplanets))) == Set([4])

        model = TerraformingAgents.galaxy_planet_setup(galaxyparams)
        planet = model.agents[1]
        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test TerraformingAgents.planet_attribute_as_matrix(candidateplanets, :pos) == 
        2 .+ [1.5  1.2  1.2
              1.5  1.0  1.2]
        ## the 2 is for "centering" the positions based on my stupid extent multiplier

        @test TerraformingAgents.planet_attribute_as_matrix(candidateplanets, :composition) == 
        [7  1  3
         7  0  3
         7  2  3]

        comp_sim_planets = TerraformingAgents.compositionally_similar_planets(planet, model; allowed_diff = 4.0)
        @test Set(map(p -> p.id, values(comp_sim_planets))) == Set([2,3])

        nearest_planets = TerraformingAgents.nearest_k_planets(planet, model, 1)
        @test Set(map(p -> p.id, values(nearest_planets))) == Set([2])

        nearest_planets = TerraformingAgents.nearest_k_planets(planet, model, 2)
        @test Set(map(p -> p.id, values(nearest_planets))) == Set([2,3])

        nearest_planets = TerraformingAgents.nearest_k_planets(planet, model, 3)
        @test Set(map(p -> p.id, values(nearest_planets))) == Set([2,3,4])

        range_planets = TerraformingAgents.planets_in_range(planet, model, 0.21)
        @test Set(map(p -> p.id, values(range_planets))) == Set([2])

        range_planets = TerraformingAgents.planets_in_range(planet, model, 0.71) 
        @test Set(map(p -> p.id, values(range_planets))) == Set([2,3,4])

        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test TerraformingAgents.most_similar_planet(planet, candidateplanets).id == 2

        candidateplanets = TerraformingAgents.basic_candidate_planets(planet, model)
        @test TerraformingAgents.nearest_planet(planet, candidateplanets).id == 2


    end
end

function TestActivateOnStepAfterSpawn()
    @testset "agents don't activate on spawn step" begin 

        galaxyparams = GalaxyParameters(
            MersenneTwister(3141),
            pos = [(1,1,1),(2,1,1),(3,1,1),(4,1,1),(5,1,1)],
            ool = 1,
            nool = 1,
            extent = (10,10,10),
            dt = 10,
            maxcomp = 1,
            compsize = 10,
            spawn_rate = 0.01,
            compmix_func=horizontal_gene_transfer,
            compmix_kwargs=Dict(:mutation_rate=>0,
                                :n_idxs_to_keep_from_destination=>1),
            compatibility_func=nearest_k_planets,
            compatibility_kwargs=Dict(:k=>10),
            destination_func=most_similar_planet)
        model = galaxy_model_setup(galaxyparams)

        @test Set(keys(model.agents)) == Set([5, 4, 6, 2, 3, 1])
        for i in 1:4
            step!(model, galaxy_agent_step_spawn_on_terraform!, galaxy_model_step!)

            i == 1 && @test Set(keys(model.agents)) == Set([5, 4, 7, 2, 3, 1])
            i == 2 && @test Set(keys(model.agents)) == Set([5, 4, 2, 8, 3, 1])
            i == 3 && @test Set(keys(model.agents)) == Set([5, 4, 2, 9, 3, 1])

        end
    end
end



@testset "All" begin
    # TestGalaxyParametersSetup()
    # TestInitializePlanetarySystems()
    # TestInitializeLife()
    # TestCompatiblePlanets()
    # TestMixCompositions()
    # TestCrossoverOnePoint()
    # TestMutation()
    # TestAgentDiesAtCorrectPlanet()
    # TestCenterPositions()
    # TestMantel()
    # TestPlanetMantelTest()
    # TestPropogationOfModelRNG()
    # TestRunningModelNoErrors()
    # TestCandidatePlanetFuncs()
    TestActivateOnStepAfterSpawn()
end