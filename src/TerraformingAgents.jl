module TerraformingAgents

using Agents, Random, AgentsPlots, Plots

Agents.random_agent(model,A::Type{T}) where {T<:AbstractAgent} = model[rand([k for (k,v) in model.agents if v isa A])]

Base.@kwdef mutable struct PlanetarySystem <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    
    nplanets::Int # To simplify the initial logic, this will be limited to 1 
    planetcompositions::Vector{Vector{Int64}} ## I'll always make 10 planet compositions, but only use the first nplanets of them
    alive::Bool
    
    ## Properties of the process, but not the planet itself
    parentplanet::Union{Int,Nothing} #id
    parentlife::Union{Int,Nothing} #id
    parentcomposition::Union{Vector{Int64},Nothing} 
    nearestlife::Union{Int,Nothing}
    
    ## Do I need the below?
    # age::Float64
    # originalcomposition::Vector{Vector{Int64}}
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    parentplanet::Int #id; this is also the "type" of life
    parentcomposition::Vector{Int64} # to simplify initial logic, this will be a single vector of length 10
    destination::Int #id of destination planetarysystem
    ## once life arrives at a new planet, life the agent just "dies"
end

function galaxy_model(; RNG::AbstractRNG = MersenneTwister(1234), speed::Float64 = 0.0, nplanets::Int =1)
    space2d = ContinuousSpace(2; periodic = true, extend = (1, 1))
    model = AgentBasedModel(Union{PlanetarySystem,Life}, space2d, properties = Dict(:dt => 1.0))

    # Add PlanetarySystem agents
    for _ in 1:10
        pos = Tuple(rand(2))
        vel = sincos(2π * rand()) .* speed
        
        psargs = Dict(:id => nextid(model),
                      :pos => pos,
                      :vel => vel)
        
        pskwargs = Dict(:nplanets => nplanets,
                        :planetcompositions => [rand(RNG,1:10,10)],
                        :alive => false,
                        :parentplanet => nothing,
                        :parentlife => nothing,
                        :parentcomposition => nothing,
                        :nearestlife => nothing)
        
        add_agent_pos!(PlanetarySystem(;merge(psargs,pskwargs)...), model)
    
    end
    
    # Add Life to a planet
    parentps = random_agent(model,PlanetarySystem)
    initialize_life(parentps, model)
    # destinationps = random_agent(model,PlanetarySystem)
    # while destinationps == parentps
    #     destinationps = random_agent(model,PlanetarySystem)
    # end
    # pos = parentps.pos
    # vel = destinationps.pos .* (speed+.1)
    
    # largs = Dict(:id => nextid(model),
    #              :pos => pos,
    #              :vel => vel)
    
    # lkwargs = Dict(:parentplanet => parentps.id,
    #                :parentcomposition => parentps.planetcompositions[1],
    #                :destination => destinationps.id)
    
    # add_agent_pos!(Life(;merge(largs,lkwargs)...), model)
        
    index!(model)
    return model
end

function initialize_life(parentps::PlanetarySystem, model::AgentBasedModel)
    destinationps = random_agent(model,PlanetarySystem)
    while destinationps == parentps
        destinationps = random_agent(model,PlanetarySystem)
    end
    pos = parentps.pos
    vel = destinationps.pos .* (speed+.1)
    
    largs = Dict(:id => nextid(model),
                 :pos => pos,
                 :vel => vel)
    
    lkwargs = Dict(:parentplanet => parentps.id,
                   :parentcomposition => parentps.planetcompositions[1],
                   :destination => destinationps.id)
    
    add_agent_pos!(Life(;merge(largs,lkwargs)...), model)
end


function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.2, :types)
        
        if typeof(a1) == PlanetarySystem
            a1.nearestlife = a2.id
        
        elseif typeof(a1) == Life
            a2.nearestlife = a1.id
        
        end
        
        # println(a2)
#         elastic_collision!(a1, a2, :mass)
    end
end

agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)

model_colors(a) = typeof(a) == PlanetarySystem ? "#2b2b33" : "#338c54"

model = galaxy_model()

e = model.space.extend
anim = @animate for i in 1:2:100
    p1 = plotabm(
        model,
        as = 5,
        ac = model_colors,
        showaxis = false,
        grid = false,
        xlims = (0, e[1]),
        ylims = (0, e[2]),
    )

    title!(p1, "step $(i)")
    step!(model, agent_step!, model_step!, 2)
end

gif(anim, "terraform_test.gif", fps = 25)

end # module
