module TerraformingAgents

using Agents, Random, AgentsPlots, Plots

Agents.random_agent(model, A::Type{T}, RNG::AbstractRNG=Random.default_rng()) where {T<:AbstractAgent} = model[rand(RNG, [k for (k,v) in model.agents if v isa A])]
# Agents.random_agent(model, A::Type{T}) where {T<:AbstractAgent} = model[rand([k for (k,v) in model.agents if v isa A])]

magnitude(x::NTuple{2,Union{Float64,Int}}) = sqrt(sum(x .^ 2))

Base.@kwdef mutable struct PlanetarySystem <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    
    nplanets::Int # To simplify the initial logic, this will be limited to 1 
    planetcompositions::Vector{Vector{Int}} ## I'll always make 10 planet compositions, but only use the first nplanets of them
    alive::Bool
    
    ## Properties of the process, but not the planet itself
    parentplanet::Union{Int,Nothing} #id
    parentlife::Union{Int,Nothing} #id
    parentcomposition::Union{Vector{Int},Nothing} 
    nearestps::Union{Int,Nothing}
    neighbors::Union{Vector{Int}}
    
    ## Do I need the below?
    # age::Float64
    # originalcomposition::Vector{Vector{Int64}}
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    parentplanet::Int #id; this is also the "type" of life
    parentcomposition::Vector{Int} # to simplify initial logic, this will be a single vector of length 10
    destination::Int #id of destination planetarysystem
    ## once life arrives at a new planet, life the agent just "dies"
end

function galaxy_model(; RNG::AbstractRNG=Random.default_rng(), psneighbor_radius::Float64 = 0.2)
    extent = (1,1)
    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = AgentBasedModel(Union{PlanetarySystem,Life}, space2d, properties = Dict(:dt => 1.0))

    initialize_planetarysystems!(model, RNG)
    initialize_psneighbors!(model, psneighbor_radius) # Add neighbor's within psneighbor_radius
    intialize_nearest_neighbor!(model, extent) # Add nearest neighbor
    # initialize_life!(random_agent(model, PlanetarySystem), model)
    initialize_life!(random_agent(model, PlanetarySystem, RNG), model)
        
    index!(model)
    return model
end

function initialize_planetarysystems!(model::AgentBasedModel, RNG::AbstractRNG = Random.default_rng())

    speed = 0.0 # for this version of initializition where all neighbors are precalculated

    # Add PlanetarySystem agents
    for _ in 1:10
        pos = Tuple(rand(RNG,2))
        vel = sincos(2π * rand(RNG)) .* speed
        
        psargs = Dict(:id => nextid(model),
                      :pos => pos,
                      :vel => vel)
        
        pskwargs = Dict(:nplanets => 1,
                        :planetcompositions => [rand(RNG,1:10,10)],
                        :alive => false,
                        :parentplanet => nothing,
                        :parentlife => nothing,
                        :parentcomposition => nothing,
                        :nearestps => nothing,
                        :neighbors => Vector{Int}(undef,0))
        
        add_agent_pos!(PlanetarySystem(;merge(psargs,pskwargs)...), model)
    
    end
end

function initialize_psneighbors!(model::AgentBasedModel, radius::Float64)
    for (a1, a2) in interacting_pairs(model, radius, :all)
        push!(a1.neighbors, a2.id)
        push!(a2.neighbors, a1.id)
        # println("psneighbors: ",a1,a2)
    end
end

function intialize_nearest_neighbor!(model::AgentBasedModel, extent::NTuple{2,Union{Float64,Int}})
    for agent in values(model.agents)
        agent.nearestps = nearest_neighbor(agent, model, magnitude(extent)).id
        # println("nearest neighbor: ",agent.id, agent.nearestps)
    end
end

function initialize_life!(parentps::PlanetarySystem, model::AgentBasedModel)
    # destinationps = random_agent(model,PlanetarySystem)
    # while destinationps == parentps
    #     destinationps = random_agent(model,PlanetarySystem)
    # end
    pos = parentps.pos
    speed = 0.2
    # vel = destinationps.pos .* 0

    println(parentps.neighbors)
    
    if length(parentps.neighbors)>0
        for neighborid in parentps.neighbors

            # println(parentps.neighbors)


            direction = (model.agents[neighborid].pos .- pos)
            direction_normed = direction ./ magnitude(direction)

            println(neighborid)
            println(model.agents[neighborid].pos)
            println(direction_normed)

            largs = Dict(:id => nextid(model),
                        :pos => pos,
                        :vel => direction_normed .* speed)
            
            lkwargs = Dict(:parentplanet => parentps.id,
                        :parentcomposition => parentps.planetcompositions[1],
                        :destination => neighborid)
            
            add_agent_pos!(Life(;merge(largs,lkwargs)...), model)
        end
    else

        direction = (model.agents[parentps.nearestps].pos .- pos)
        direction_normed = direction ./ magnitude(direction)

        largs = Dict(:id => nextid(model),
                    :pos => pos,
                    :vel => direction_normed .* speed)
            
        lkwargs = Dict(:parentplanet => parentps.id,
                    :parentcomposition => parentps.planetcompositions[1],
                    :destination => parentps.nearestps)
        
        add_agent_pos!(Life(;merge(largs,lkwargs)...), model)

    end


end


# function model_step!(model)
#     for (a1, a2) in interacting_pairs(model, 0.2, :types)
        
#         if typeof(a1) == PlanetarySystem
#             a1.nearestlife = a2.id
        
#         elseif typeof(a1) == Life
#             a2.nearestlife = a1.id
        
#         end
        
#         # println(a2)
# #         elastic_collision!(a1, a2, :mass)
#     end
# end

agent_step!(agent, model) = move_agent!(agent, model, model.dt/10)

modelparams = Dict(:RNG => MersenneTwister(1236),
                   :psneighbor_radius => .45)

model = galaxy_model(;modelparams...)

model_colors(a) = typeof(a) == PlanetarySystem ? "#2b2b33" : "#338c54"

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
    step!(model, agent_step!, 2) # model_step!, 2)
end


animation_path = "../output/animation/"
if !ispath(animation_path)
    mkpath(animation_path)
end

gif(anim, joinpath(animation_path,"terraform_test4.gif"), fps = 25)

end # module

#### Do all the calculations for nearest neighbors at the begining if the planetary systems don't move_agent
# - Otherwise, do them at each step if they do move
# - don't go back to planet you came from or planet that already has your life on it
# - fix velocity so that you go every direction at same speed and doesn't depend on how far away your target is. 
