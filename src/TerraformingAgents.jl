module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict

Agents.random_agent(model, A::Type{T}, RNG::AbstractRNG=Random.default_rng()) where {T<:AbstractAgent} = model[rand(RNG, [k for (k,v) in model.agents if v isa A])]
# Agents.random_agent(model, A::Type{T}) where {T<:AbstractAgent} = model[rand([k for (k,v) in model.agents if v isa A])]

magnitude(x::Tuple{<:Real,<:Real}) = sqrt(sum(x .^ 2))

Base.@kwdef mutable struct PlanetarySystem <: AbstractAgent
    id::Int
    pos::Tuple{<:Real,<:Real}
    vel::Tuple{<:Real,<:Real}
    
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
    pos::Tuple{<:Real,<:Real}
    vel::Tuple{<:Real,<:Real}
    parentplanet::Int #id; this is also the "type" of life
    parentcomposition::Vector{Int} # to simplify initial logic, this will be a single vector of length 10
    destination::Int #id of destination planetarysystem
    ## once life arrives at a new planet, life the agent just "dies"
end

function galaxy_model(
    nplanetarysystems::Int ; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    psneighbor_radius::Real = 0.2, ## distance threshold used to decide where to send life from parent planet
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    similarity_threshold::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    nplanetspersystem::Int = 1) ## number of planets per star)

    pos = [Tuple(rand(RNG,2)) for _ in 1:nplanetarysystems]
    vel = [(0,0) for _ in 1:nplanetarysystems]
    planetcompositions = [[rand(RNG,1:10,nplanetspersystem)] for _ in 1:nplanetarysystems]

end

function galaxy_model(; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    psneighbor_radius::Real = 0.2, ## distance threshold used to decide where to send life from parent planet
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    similarity_threshold::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    nplanetspersystem::Int = 1, ## number of planets per star
    pos::Union{Nothing,AbstractArray{Tuple{<:Real,<:Real}}} = nothing,
    vel::AbstractArray{Tuple{<:Real,<:Real}} = nothing,
    planetcompositions::Vector{Vector{Vector{Int}}} = nothing)

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = AgentBasedModel(
        Union{PlanetarySystem,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            similarity_threshold, 
            psneighbor_radius))

    initialize_planetarysystems!(model, RNG, nplanetarysystems, nplanetspersystem, pos, vel, planetcompositions)
    initialize_psneighbors!(model, psneighbor_radius) # Add neighbor's within psneighbor_radius
    intialize_nearest_neighbor!(model, extent) # Add nearest neighbor
    initialize_life!(random_agent(model, PlanetarySystem, RNG), model)   
    index!(model)
    
    return model

end

# isargprovided(args::NamedTuple) = all(map(isnothing,args)) && ArgumentError("one of $(keys(args)) must be provided")

function providedargs(args::Dict) 
    
    providedargs = filter(x -> !isnothing(x.second), args)

    isempty(providedargs) ? throw(ArgumentError("one of $(keys(args)) must be provided")) : providedargs

end 

haveidenticallengths(args::Dict) = all(length(i.second) == length(args[collect(keys(args))[1]]) for i in args)



# function haveidenticallengths(args::NamedTuple)

#     providedargs = collect(args)[collect(args) .!== nothing] 

#     if all(map(isnothing,args))
#         throw(ArgumentError("one of $(keys(args)) must be provided"))
#     elseif ~all(length(i) == length(providedargs[1]) for i in providedargs)
#         throw(ArgumentError("provided arguments $(keys(args)[collect(args) .!== nothing]) must all be same length"))
#     else
#         return true
#     end

# end

# function initialize_planetarysystems!(
#     model::AgentBasedModel,
#     nplanetarysystems::Int = 10; 
#     RNG::AbstractRNG = Random.default_rng(),
#     nplanetspersystem::Int = 1)

#     println("nothing")

# end

function initialize_planetarysystems!(
    model::AgentBasedModel; 
    RNG::AbstractRNG = Random.default_rng(),
    nplanetarysystems::Int = 10,
    nplanetspersystem::Int = 1,
    pos::Union{Nothing,AbstractArray{Tuple{<:Real,<:Real}}} = nothing,
    vel::AbstractArray{Tuple{<:Real,<:Real}} = nothing,
    planetcompositions::Vector{Vector{Vector{Int}}} = nothing)
    ##  MAYBE I SHOULD SWITCH PLANETCOMPOSITIONS TO BEING ABSTRACT ARRAY? BUT THEN TESTING GETS HARDER, BUT ALLOWS EASIER TRANSITION TO STATIC IN THE FUTURE
    ## COULD DO THIS A BETTER WAY WITH MULTIPLE DISPATCH SO THAT I DON'T ALLOW USER TO OVERCONSTRAIN

    ## Ensure at least one of the following args is provided
    args = Dict(pos=>pos, vel=>vel, planetcompositions=>planetcompositions)
    ## Get user provided args
    userargs = providedargs(args)
    ## Verify userargs aren't overconstrained
    haveidenticallength(userargs) || throw(ArgumentError("provided arguments $(keys(userargs)) must all be same length"))


    nplanetarysystems = length(args[args .!== nothing][1])

    ## Initialize arguments which are not provided 
    ## (flat random pos, no velocity, flat random compositions)
    isnothing(pos) && (pos = [Tuple(rand(RNG,2)) for _ in 1:nplanetarysystems])
    isnothing(vel) && (vel = [(0,0) for _ in 1:nplanetarysystems])
    isnothing(planetcompositions) && ([[rand(RNG,1:10,nplanetspersystem)] for _ in 1:nplanetarysystems])

    if isnothing(nplanetspersystem) & isnothing(planetcompositions)
        nplanetspersystem = 1
        planetcompositions = [[rand(RNG,1:10,nplanetspersystem)] for _ in 1:nplanetarysystems]
    elseif isnothing(planetcompositions)
        planetcompositions = [[rand(RNG,1:10,nplanetspersystem)] for _ in 1:nplanetarysystems]
    elseif isnothing(nplanetspersystem)
        ## pass-- this effectively allows for specifiying heterogenous numbers of planets per star
        ## already checked in hasidenticallength whether planetcompositions has the right length 
    elseif length(planetcompositions) != nplanetspersystem
        # NEED TO CHANGE THIS TO CHECK THE LGNTH OF EACH VECTOR IN PLANETCOMPOSITONS
        ArgumentError("Model is overconstrained: length(planetcompositions) must equal nplanetspersystem")
    else ## nplanetspersystem and planetcompositions are both specified
         ## this is ok if they're the 
    end
        


    for composition in planetcompositions
        length(composition) != nplanetspersystem && ArgumentError("length(planetcompositions[i]) must equal nplanetspersystem for all i")
    end
    # speed = 0.0 # for this version of initializition where all neighbors are precalculated

    # Add PlanetarySystem agents
    for i in 1:nplanetarysystems
        # pos = Tuple(rand(RNG,2))
        # vel = sincos(2π * rand(RNG)) .* speed
        
        pskwargs = Dict(:id => nextid(model),
                    :pos => pos[i],
                    :vel => vel[i],
                    :nplanets => nplanetspersystem,
                    :planetcompositions => [rand(RNG,1:10,10)],
                    :alive => false,
                    :parentplanet => nothing,
                    :parentlife => nothing,
                    :parentcomposition => nothing,
                    :nearestps => nothing,
                    :neighbors => Vector{Int}(undef,0))
        
        add_agent_pos!(PlanetarySystem(;pskwargs...), model)
    
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

    # println(parentps.neighbors)
    
    if length(parentps.neighbors)>0
        for neighborid in parentps.neighbors

            # println(parentps.neighbors)


            direction = (model.agents[neighborid].pos .- pos)
            direction_normed = direction ./ magnitude(direction)

            # println(neighborid)
            # println(model.agents[neighborid].pos)
            # println(direction_normed)

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

function is_compatible(life::Life, ps::PlanetarySystem, threshold::Float64)

    ## Placeholder
    return true

end

function terraform!(life::Life, ps::PlanetarySystem)
    ps.alive = true
    ps.parentplanet = life.parentplanet
    ps.parentcomposition = life.parentcomposition
    println("terraformed $(ps.id) from $(life.id)")
    ## Change ps composition based on life parent composition and current planet composition
    # ps.planetcompositions[1] = mix_compositions(life, ps)

end

function galaxy_model_step!(model)
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, ps = typeof(a1) == PlanetarySystem ? (a2, a1) : (a1, a2)
        is_compatible(life, ps, model.similarity_threshold) ? terraform!(life, ps) : return
        kill_agent!(life, model)
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

# function sir_agent_step!(agent, model)
#     move_agent!(agent, model, model.dt)
#     update!(agent) # store information in life agent of it terraforming?
#     recover_or_die!(agent, model)
# end

## COMMENTING OUT EVEYRTHING BELOW FOR TESTING

# agent_step!(agent, model) = move_agent!(agent, model, model.dt)

# modelparams = Dict(:RNG => MersenneTwister(1236),
#                    :psneighbor_radius => .45,
#                    :dt => 0.1)

# model = galaxy_model(;modelparams...)

# model_colors(a) = typeof(a) == PlanetarySystem ? "#2b2b33" : "#338c54"

# e = model.space.extend
# anim = @animate for i in 1:2:100
#     p1 = plotabm(
#         model,
#         as = 5,
#         ac = model_colors,
#         showaxis = false,
#         grid = false,
#         xlims = (0, e[1]),
#         ylims = (0, e[2]),
#     )

#     title!(p1, "step $(i)")
#     step!(model, agent_step!, galaxy_model_step!, 2)
# end


# animation_path = "../output/animation/"
# if !ispath(animation_path)
#     mkpath(animation_path)
# end

# gif(anim, joinpath(animation_path,"terraform_test_death1.gif"), fps = 25)

end # module

#### Do all the calculations for nearest neighbors at the begining if the planetary systems don't move_agent
# - Otherwise, do them at each step if they do move
# - don't go back to planet you came from or planet that already has your life on it
# - fix velocity so that you go every direction at same speed and doesn't depend on how far away your target is. 
