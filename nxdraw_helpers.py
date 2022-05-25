from matplotlib.cm import viridis
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap, LogNorm
from matplotlib.collections import LineCollection
import numpy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from flownetpy.tools import FlowDict


def calculate_edge_offset(edge_pos, offset):
    edge_dy = edge_pos[:,1,1] -edge_pos[:,0,1]
    edge_dx = edge_pos[:,1,0] -edge_pos[:,0,0]

    edge_angles = np.where(np.isclose(edge_dx,0), np.pi/2, np.arctan(edge_dy/edge_dx))

    yoffsets = offset*np.cos(edge_angles)
    xoffsets = offset*np.sin(edge_angles)

    offsets = np.array([xoffsets, yoffsets]).T

    return np.swapaxes(np.swapaxes(edge_pos, 0,1)+offsets, 0,1)


def annotate_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=None,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        label=None,
                        offset = 0,
                        **kwds):
    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    # set edge positions
    edge_pos = numpy.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    edge_pos = calculate_edge_offset(edge_pos, offset)
    
    if not hasattr(width, '__iter__'):
        lw = (width,)
    else:
        lw = width

    if not cb.is_string_like(edge_color) \
           and hasattr(edge_color, '__iter__') \
           and len(edge_color) == len(edge_pos):
        if numpy.alltrue([cb.is_string_like(c)
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif numpy.alltrue([not cb.is_string_like(c)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if numpy.alltrue([cb.iterable(c) and len(c) in (3, 4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if cb.is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')

    edge_collection = LineCollection(edge_pos,
                                     colors=edge_colors,
                                     linewidths=lw,
                                     antialiaseds=(1,),
                                     linestyle=style,
                                     transOffset = ax.transData,
                                     )

    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)

    # Note: there was a bug in mpl regarding the handling of alpha values for
    # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
    # r7189 (June 6 2009).  We should then not set the alpha value globally,
    # since the user can instead provide per-edge alphas now.  Only set it
    # globally if provided as a scalar.
    if cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(numpy.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    arrow_collection = None

    if G.is_directed() and arrows:

        # a directed graph hack
        # draw thick line segments at head end of edge
        # waiting for someone else to implement arrows that will work
        arrow_colors = edge_colors
        a_pos = []
        p = 1.0-0.25  # make head segment 25 percent of edge length
        for src, dst in edge_pos:
            x1, y1 = src
            x2, y2 = dst
            dx = x2-x1   # x offset
            dy = y2-y1   # y offset
            d = numpy.sqrt(float(dx**2 + dy**2))  # length of edge
            if d == 0:   # source and target at same position
                continue
            if dx == 0:  # vertical edge
                xa = x2
                ya = dy*p+y1
            if dy == 0:  # horizontal edge
                ya = y2
                xa = dx*p+x1
            else:
                theta = numpy.arctan2(dy, dx)
                xa = p*d*numpy.cos(theta)+x1
                ya = p*d*numpy.sin(theta)+y1

            a_pos.append(((xa, ya), (x2, y2)))

        arrow_collection = LineCollection(a_pos,
                                colors=arrow_colors,
                                linewidths=[4*ww for ww in lw],
                                antialiaseds=(1,),
                                transOffset = ax.transData,
                                )

        arrow_collection.set_zorder(1)  # edges go behind nodes
        arrow_collection.set_label(label)
        ax.add_collection(arrow_collection)

    # update view
    minx = numpy.amin(numpy.ravel(edge_pos[:, :, 0]))
    maxx = numpy.amax(numpy.ravel(edge_pos[:, :, 0]))
    miny = numpy.amin(numpy.ravel(edge_pos[:, :, 1]))
    maxy = numpy.amax(numpy.ravel(edge_pos[:, :, 1]))

    w = maxx-minx
    h = maxy-miny
    padx,  pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

#    if arrow_collection:

    return edge_collection


def draw_edge_arrowheads(G, pos, flows, axis):
    """Plots arrpw heads on a networkx graph"""
    for u,v in G.edges():
        if flows[(u,v)]>0:
            a, b = u,v
        else:
            a, b = v,u

        ax,ay = pos[a]
        bx,by = pos[b]

        axis.arrow(ax, ay, (bx-ax)/2.4, (by-ay)/2.4, head_width=0.2, head_length=0.1, fc="0", ec='w', width=0, lw=0)



def draw_networkx_coloured_by_flow(G, flows, inputs=None, ax=None, pos=None,
                                   nodelist=None, edgelist=None, with_cbar=True,
                                   cbar_label=None, vmin=None, vmax=None, cmap=None,
                                   **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal')

    if pos is None:
        pos = nx.spring_layout(G)

    if not hasattr(flows, 'keys'):
        #its a list, convert to dict
        flows = to_flowdict(flows, G)

    # draw source and sink nodes with different shapes
#    draw_networkx_src_and_sink(G, inputs, pos, ax)
    # now the edges colour coded by flow
    abs_flows_arr = [abs(flows[edge]) for edge in G.edges_arr]
    ax, cbar = draw_networkx_colored_edges(G, abs_flows_arr, ax=ax, pos=pos, with_cbar=with_cbar,
                                cbar_label=cbar_label, vmin=vmin, vmax=vmax, cmap=cmap,**kwargs)
    # now plot the arrowheads
    draw_edge_arrowheads(G, pos, flows, ax)
    return ax, cbar


def draw_networkx_src_and_sink(G, inputs=None, pos=None, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal')

    if pos is None:
        pos = nx.spring_layout(G)


    if inputs is not None:
        # user supplied inputs, overrides Graphs node attr
        for node in G.nodes():
            G.node[node]['input'] = inputs[node]

    gen_nodes = [node for node in G.nodes() if
                        G.node[node]['input']>0]
    con_nodes = [node for node in G.nodes() if
                        G.node[node]['input']<0]

    npos=nx.draw_networkx_nodes(G, ax = ax, pos=pos, nodelist = gen_nodes, alpha=0.8,
                          node_size=100, vmin=-1, vmax=1, cmap = 'viridis')

    nneg=nx.draw_networkx_nodes(G, ax = ax, pos=pos, nodelist = con_nodes, alpha=0.8,
                          node_size=100, vmin=-1, vmax=1, cmap='viridis', node_shape='s')

    xs,ys = [i[0] for i in pos.values()], [i[1] for i in pos.values()]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    ax.set_xlim(min(xs)-0.05*width, max(xs)+0.05*width)
    ax.set_ylim(min(ys)-0.05*height, max(ys)+0.05*height)
    ax.set_frame_on(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)



def draw_networkx_colored_edges(G, colors, edgelist=None, ax = None, pos = None,
                                cmap = None, vmin=None, vmax=None,
                                with_cbar = True, cbar_orient = 'right', cbar_label=None,
                                width=5,
                                **kwargs):
    """Draws a graph with edges colored acc to flows (determined from flows)"""
    if pos is None:
        pos=nx.circular_layout(G)

    if edgelist is None:
        edgelist = G.edges_arr

    if cmap is None:
        cmap = matplotlib.cm.viridis

    if ax is None:
        ax = plt.gca()

    e=nx.draw_networkx_edges(G, ax = ax, pos=pos, edgelist = edgelist, 
                             edge_color = colors, edge_cmap = cmap,
                             edge_vmin = vmin, width = width, edge_vmax = vmax, **kwargs)

    if with_cbar:
        if cbar_orient == 'right':
            #cax = ax.get_figure().add_axes([1,0.1,0.05,0.4])
            #cbar=plt.gcf().colorbar(e, ax=ax, cax = cax, use_gridspec=True)
            cbar=plt.gcf().colorbar(e, ax=ax, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=30)
            #cax.yaxis.tick_left()
            #cax.yaxis.set_label_position('right')

        elif cbar_orient == 'left':
            cax = ax.get_figure().add_axes([-0.1,0.1,0.05,0.4])
            cbar=plt.colorbar(e, ax=ax, cax = cax, use_gridspec=True)
            cbar.set_label(cbar_label, fontsize=30)
            cax.yaxis.set_label_position('left')

        elif cbar_orient == 'top':
            cax = ax.get_figure().add_axes([0,1.1,0.05,0.8])
        else:
            pass

        cbar.ax.tick_params(labelsize=30) 

    ax.set_frame_on(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    if with_cbar:
        return ax, cbar
    else:
        return ax, None

def create_patches(positions, angles):
    angles = angles - angles[0]
    angles = [angle*180/np.pi for angle in angles]
    print(angles)
    ps = [mpatches.Wedge(pos, 0.06, min(0, angle), max(0, angle), fill = True,\
                         ec = "none")\
         for pos, angle in zip(positions, angles)]

    cs = [mpatches.Wedge(pos, 0.06, 0, 360, fill = False,\
                         ec = "k", lw = 1)\
         for pos, angle in zip(positions, angles)]
    return ps + cs



def plot_polar_angles(thetas):
    """Draws thetas in a polar plot to demonstrate winding numbers"""
    ax = plt.subplot(111, polar=True)
    r = np.arange(len(thetas))
    ax.plot(thetas - thetas[0], r, 'bo-', label="Before bifurcation")
 
    ax.get_yaxis().set_ticks([])
    plt.legend(fontsize=20, bbox_to_anchor=(1.9, 1.05))

    # Set up radian ticks
    xT=plt.xticks()[0]
    xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    plt.xticks(xT, xL)