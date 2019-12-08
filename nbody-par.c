/*  
    Vasileios Gkanasoulis
    Parallel N-Body simulation code.

    The bodies N are divided amongst the participating nodes P (approximately N/P per node).
    As the assignment declaration states, the force computation is the only part that requires
    communication. During that step, the only information that is essential for computations is
    the positions of the bodies. Therefor I created another struct for that purpose since sending
    the whole body = 10 x 8 bytes and the positions are 2 x 8 bytes (essentially reducing message size).

    Now onto the most interesting part: how and when the nodes communicated to exchange body positions?
    Initially I thought of using something like MPI_Gather on each node to collect all of the positions
    from the rest of the nodes. Though I soon realised that this way will not scale well as the # of bodies
    increases. This is because each node would require 8 bytes x (N - N/P) of data in each step and during 
    the time that the communication takes place all nodes have to partake (not compute anything).

    As a better solution I overlapped computations with communication, by using asynchronous communication
    and creating a communication pipeline.
    Short explanation:
    Assuming 4 nodes are used, each having a distinct set of bodies (let's call them sets A, B, C, D)
    The computations are done in steps where # steps == # of nodes, with each step computing forces of
    one set to another.
    The pipeline steps:

    Step 0:    A with A     B with B      C with C    D with D    (step 0 is the interactions of local bodies)
    Step 1:    D with A  -> A with B ->   B with C -> C with D 
    Step 2:    C with A  -> D with B |->| A with C -> B with D 
    Step 3:    B with A  -> C with B ->   D with C -> A with D 

    Before Step 0, each process calls MPI_Irecv / MPI_Isend sending the data that will be required in
    Step 1 to the respective neighbours. After local computations, MPI_Waiall is called to ensure that
    the async communication is completed. The same overlapping occurs with the rest of the steps.

    As an additional optimisation I also reduced the number of pipeline steps required as the above
    explanation does not take advantage of symmetry (force of body 1 on 2 is the negative of 2 on 1).
    Essentially everything after the |->| notation in the graphical representation is the opposite
    forces of interactions allready computed. Therefore this implementation skips everything after
    |->| (The negative forces are computed in the previous steps). Since the total number of steps
    is equal to the number of nodes used, this optimisation almost cuts in half the number of
    steps required. Finally, after the steps are completed all nodes call MPI_Allreduce to get all
    of the forces that correspond to their local bodies.
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>//for offsetof
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>


#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015
#define ROOT        0

MPI_Status statuses[2], o_sts[2];
MPI_Request requests[2], o_req[2];

int rank, nodes, half_nodes, rmt_nodes, end_step;
int leftover_bodies, bodies_per_node, max_bodies_per_node;
int left_neighbour, right_neighbour;
int *s_r_counts, *displacements, *r_neighbours;
double *g_forces_x, *g_forces_y;
bool uneven_dist=false, single_node=false, split=false, symm_part=false;

/*Custom MPI datatype for scattering/gathering bodies*/
MPI_Datatype mpiBodyType;
MPI_Datatype types[8] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
int typelengths[8] = {2,2,1,1,1,1,1,1};
MPI_Aint disp[8];

/*Custom MPI datatype for sending/receiving coordinates information*/
MPI_Datatype mpi_bodyPositions;
MPI_Datatype exchange_types[2] = {MPI_DOUBLE,MPI_DOUBLE};
int exchange_lengths[2] = {1,1};
MPI_Aint disp_loc[2];

typedef struct bodyType {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
}bodyType;


/*Struct containig body info that is essential during the compute_forces() function.
  Since Mass and radius are constant and velocities are computed independently,
  there is no need to send the whole body.*/

struct bodyPositions {
    double x;
    double y;
};


struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1
    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};


struct localworld{
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    
    int                 xdim;
    int                 ydim;
};


/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies[B].x[(w)->old]
#define XN(w, B)       (w)->bodies[B].x[(w)->old^1]
#define Y(w, B)        (w)->bodies[B].y[(w)->old]
#define YN(w, B)       (w)->bodies[B].y[(w)->old^1]
#define XF(w, B)       (w)->bodies[B].xf
#define YF(w, B)       (w)->bodies[B].yf
#define XV(w, B)       (w)->bodies[B].xv
#define YV(w, B)       (w)->bodies[B].yv
#define R(w, B)        (w)->bodies[B].radius
#define M(w, B)        (w)->bodies[B].mass


static inline void
clear_forces(struct localworld *world)
{
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < world->bodyCt; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}


/*Forces computations*/
static inline void compute_forces_mpi_symm(struct localworld * restrict l_world, struct world * restrict world,
 struct bodyPositions *restrict rcvBodies, struct bodyPositions * restrict sendBodies, int steps,
  int h_steps, int n_idx){

    int step, b, c;
    int c_l_n_idx, f_l_n_idx, f_r_n_idx;

    /*First step is on local bodies: A on A*/
    /*Overlap communication with computations*/
    MPI_Irecv(rcvBodies, s_r_counts[left_neighbour], mpi_bodyPositions, left_neighbour, 1, MPI_COMM_WORLD, &(requests[0]));
    MPI_Isend(sendBodies, s_r_counts[rank], mpi_bodyPositions, right_neighbour, 1, MPI_COMM_WORLD, &(requests[1]));

    for (b = 0; b < l_world->bodyCt; ++b) {
        for (c = b + 1; c < l_world->bodyCt; ++c) {
            double dx = X(l_world, c) - X(l_world, b);
            double dy = Y(l_world, c) - Y(l_world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(l_world, b) + R(l_world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(l_world, b) * M(l_world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            XF(l_world, b) += xf;
            YF(l_world, b) += yf;
            XF(l_world, c) -= xf;
            YF(l_world, c) -= yf;
        }
    }

    MPI_Waitall(2, requests, statuses); //Ensure messages arrived

    for (step = 1; step < steps; ++step){
        /*Figure out neighbours for the next(future) step, not the one we are about to compute*/
        f_l_n_idx = r_neighbours[n_idx - 1 - step]; 
        f_r_n_idx = r_neighbours[n_idx + 1 + step];
        c_l_n_idx = r_neighbours[n_idx - step]; 

        /*The following denotes when nodes need to stop computing*/
        if(step > h_steps || (split && rank >= half_nodes && step == h_steps)) symm_part = true;


        if (!(step > h_steps)){ //Stop communicating halfway through the pipeline steps
            MPI_Isend(sendBodies, s_r_counts[rank], mpi_bodyPositions, f_r_n_idx, 2, MPI_COMM_WORLD, &(o_req[0]));
        }

        if(!symm_part){//Dont compute if it is not needed
            for(b = 0; b < s_r_counts[c_l_n_idx]; ++b){
                for(c = 0; c < l_world->bodyCt; ++c){
                    int world_idx = displacements[c_l_n_idx] + b;
                    double dx =  rcvBodies[b].x -  X(l_world, c);
                    double dy =  rcvBodies[b].y - Y(l_world, c);
                    double angle = atan2(dy, dx);
                    double dsqr = dx*dx + dy*dy;
                    double mindist = R(l_world, c) + R(world, world_idx);
                    double mindsqr = mindist*mindist;
                    double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
                    double force = M(l_world, c) * M(world, world_idx) * GRAVITY / forced;
                    double xf = force * cos(angle);
                    double yf = force * sin(angle); 

                    XF(l_world, c) += xf;
                    YF(l_world, c) += yf;
                    g_forces_x[world_idx] -= xf; //Store opposite forces
                    g_forces_y[world_idx] -= yf;
                }
            }
        }

        if (!(step > h_steps)){//Finally receive the positions for the next stage
            MPI_Irecv(rcvBodies, s_r_counts[f_l_n_idx], mpi_bodyPositions, f_l_n_idx, 2, MPI_COMM_WORLD, &(o_req[1]));
            MPI_Waitall(2, o_req, o_sts);
        }  
    }//End pipeline

    symm_part = false;
    /*Forces have been computed. now reduce the counter forces that were stored locally*/
    MPI_Allreduce(MPI_IN_PLACE, g_forces_x, world->bodyCt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, g_forces_y, world->bodyCt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /*Accumulate reduced forces on local bodies*/
    for (int l = 0; l < l_world->bodyCt; ++l){
        int global_idx = displacements[rank] + l;
        XF(l_world, l) += g_forces_x[global_idx];
        YF(l_world, l) += g_forces_y[global_idx];
    }
}


static inline void
compute_velocities(struct localworld *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;
    }
}

static inline void
compute_positions(struct localworld *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;
    }
}


/*  Graphic output stuff...
 */

#include <fcntl.h>
#include <sys/mman.h>

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void
filemap_close(struct filemap *filemap)
{
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *
Eat_Space(unsigned char *p)
{
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }
    return p;
}


static unsigned char *
Get_Number(unsigned char *p, int *n)
{
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int
map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap)
{
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }

    /* File should now be mapped; read magic value */
    p = filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;

    return 0;

ppm_abort:
    filemap_close(filemap);

    return -1;
}


static inline void
color(const struct world *world, unsigned char *image, int x, int y, int b)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));
    int tint = ((0xfff * (b + 1)) / (world->bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(const struct world *world, unsigned char *image, int x, int y)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

static void
display(const struct world *world, unsigned char *image)
{
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < world->ydim; ++j) {
        for (i = 0; i < world->xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < world->bodyCt; ++b) {
                double dy = Y(world, b) - j;
                double dx = X(world, b) - i;
                double d = sqrt(dx*dx + dy*dy);

                if (d <= R(world, b)+0.5) {
                    /* This is it */
                    color(world, image, i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(world, image, i, j);

colored:        ;
        }
    }
}

static void
print(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
    }
}

/*MISCELLANEOUS FUNCTIONS*/
/*Figure out how many bodies to assign per node. Also calculate send/receive and displacement counts*/
static inline void 
divide_bodies_to_nodes(struct world *world){
    leftover_bodies = world->bodyCt % nodes;
    /*Buffers*/
    s_r_counts = (int*) malloc(sizeof(int) * nodes);
    displacements = (int*) malloc(sizeof(int) * nodes);
    r_neighbours = (int*) malloc(sizeof(int) * nodes * 3);

    if (s_r_counts == NULL || displacements == NULL || r_neighbours == NULL){
        fprintf(stderr, "Cannot malloc buffers in divide_bodies_to_nodes()\n");
        exit(1);  
    }

    g_forces_x = (double*) malloc(sizeof(double) * world->bodyCt);
    g_forces_y = (double*) malloc(sizeof(double) * world->bodyCt);

    if ( g_forces_x == NULL || g_forces_y == NULL){
        fprintf(stderr, "Cannot malloc force buffers in divide_bodies_to_nodes()\n");
        exit(1);   
    }

    int temp_bodies_pn = (int) world->bodyCt / nodes;
    int temp_n = 0;
    displacements[0] = 0;

    if(leftover_bodies == 0){
        bodies_per_node = world->bodyCt / nodes;
    } else {
        uneven_dist=true;
        bodies_per_node = (int) world->bodyCt / nodes;
        bodies_per_node = (rank < leftover_bodies) ? bodies_per_node+1 : bodies_per_node;
    }

    max_bodies_per_node = (leftover_bodies == 0) ? bodies_per_node : bodies_per_node + 1;
    for (int i=0; i<nodes; ++i){
        s_r_counts[i] = (i < leftover_bodies)? temp_bodies_pn+1 : temp_bodies_pn;
    }

    for (int d=1; d<nodes; ++d){
        displacements[d] =  (d-1 < leftover_bodies)? temp_bodies_pn+1 : temp_bodies_pn;
        displacements[d] += displacements[d-1];
    }
    /*Contains replicated ranks for rotation of neighbours in communication pipeline: 3 nodes= 0 1 2 0 1 2 0 1 2*/
    for (int t = 0; t < 3; ++t){
        for (int n = 0; n < nodes; ++n){
            r_neighbours[temp_n + n] =  n;
        }
        temp_n += nodes;
    }
}

/*For clearing the forces in buffer at each step*/
static inline void zero_force_buffers(int count){
    for(int f = 0; f < count; ++f){
        g_forces_x[f] = 0;
        g_forces_y[f] = 0;
    }
}

static void set_neighbours(){
    left_neighbour = (rank == ROOT) ? nodes - 1 : rank - 1 ; 
    right_neighbour = (rank == nodes - 1) ? ROOT : rank + 1 ;
}


static inline void fill_send_buff(struct localworld * restrict world, struct bodyPositions * restrict sendBodies){
    for (int f = 0; f < world->bodyCt; ++f){
        sendBodies[f].x = X(world, f);
        sendBodies[f].y = Y(world, f);
    }
}

static int is_even(int x){
    return !(x & 1);
}



/*  Main program...*/
int
main(int argc, char **argv)
{
    unsigned int lastup = 0;
    unsigned int secsup;
    int b;
    int steps, n_arr_idx;
    double start_t, end_t;    

    struct filemap image_map;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);

    /*Figure out pipeline information*/
    if (nodes-1 == ROOT) single_node = true;
    if (!(is_even(nodes-1))) split = true;
    half_nodes = nodes / 2;
    end_step = nodes - 1;
    end_step = (end_step + 2 -1) / 2;

    struct world *world = calloc(1, sizeof *world);
    struct localworld *l_world = calloc(1, sizeof *l_world);


    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }
    if (l_world == NULL) {
        fprintf(stderr, "Cannot calloc(l_world)\n");
        exit(1);
    }


    /* Get Parameters */
    if (argc != 5) {
        fprintf(stderr, "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;
    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }
    secsup = atoi(argv[2]);
    if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }
    steps = atoi(argv[4]);

    if(world->bodyCt < nodes){
        fprintf(stderr, "Bodies are less than specified nodes, setting bodies to %i\n", nodes);
        world->bodyCt = nodes;
    } 


    if (rank == ROOT) fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);

    /* Initialize simulation data on ROOT*/
    srand(SEED);
        for (b = 0; b < world->bodyCt; ++b) {
            X(world, b) = (rand() % world->xdim);
            Y(world, b) = (rand() % world->ydim);
            R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
            M(world, b) = R(world, b) * R(world, b) * R(world, b);
            XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
            YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        }

    divide_bodies_to_nodes(world);
    set_neighbours();
    n_arr_idx = nodes + rank;
    /*structs for send/rcv*/
    struct bodyPositions sendBodies[max_bodies_per_node];
    struct bodyPositions receiveBodies[max_bodies_per_node];

    zero_force_buffers(world->bodyCt);

    /*Create mpi struct for scatter/gather*/
    disp[0] = offsetof(bodyType, x);
    disp[1] = offsetof(bodyType, y);
    disp[2] = offsetof(bodyType, xf);
    disp[3] = offsetof(bodyType, yf);
    disp[4] = offsetof(bodyType, xv);
    disp[5] = offsetof(bodyType, yv);
    disp[6] = offsetof(bodyType, mass);
    disp[7] = offsetof(bodyType, radius);

    MPI_Type_create_struct(8, typelengths, disp, types, &mpiBodyType);
    MPI_Type_commit(&mpiBodyType);

    /*create mpi struct for send/receive*/
    disp_loc[0] = offsetof(struct bodyPositions, x);
    disp_loc[1] = offsetof(struct bodyPositions, y);

    MPI_Type_create_struct(2, exchange_lengths, disp_loc, exchange_types, &mpi_bodyPositions);
    MPI_Type_commit(&mpi_bodyPositions);

    /*Distribute bodies*/
    if (!uneven_dist){
        if (rank==ROOT){
            MPI_Scatter(world->bodies, s_r_counts[rank], mpiBodyType, l_world->bodies, s_r_counts[rank], mpiBodyType, ROOT, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(NULL, 0, mpiBodyType, l_world->bodies, s_r_counts[rank], mpiBodyType, ROOT, MPI_COMM_WORLD);
        }
    } else {
        if (rank==ROOT){
            MPI_Scatterv(world->bodies, s_r_counts, displacements, mpiBodyType, l_world->bodies, s_r_counts[rank], mpiBodyType, ROOT, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(NULL, s_r_counts, displacements, mpiBodyType, l_world->bodies,s_r_counts[rank], mpiBodyType, ROOT, MPI_COMM_WORLD);
        }
    }


    l_world->bodyCt = bodies_per_node;
    l_world->xdim = world->xdim;
    l_world->ydim = world->ydim;    

    start_t = MPI_Wtime();

    /* Main Loop */
    while (steps--) {
        clear_forces(l_world);
        zero_force_buffers(world->bodyCt);//reset
        fill_send_buff(l_world, sendBodies);//prepare bodies to be send 
        compute_forces_mpi_symm(l_world, world, receiveBodies, sendBodies, nodes, end_step, n_arr_idx);
        compute_velocities(l_world);
        compute_positions(l_world);
        
        /* Flip old & new coordinates */
        l_world->old ^= 1;

        /*Time for a display update?*/ 
        if (secsup > 0 && (time(0) - lastup) > secsup) {
            display(world, image_map.image);
            msync(image_map.map, image_map.fsize, MS_SYNC); /* Force write */
            lastup = time(0);
        }
    }


    end_t = MPI_Wtime();

    /*Gather back results*/
    if (!uneven_dist){
        if (rank==ROOT){
            MPI_Gather(l_world->bodies, s_r_counts[rank], mpiBodyType, world->bodies, s_r_counts[rank], mpiBodyType,ROOT,MPI_COMM_WORLD);
        } else {
             MPI_Gather(l_world->bodies, s_r_counts[rank], mpiBodyType, NULL, 0, mpiBodyType,ROOT,MPI_COMM_WORLD);
        }
    } else {
        if (rank==ROOT){
            MPI_Gatherv(l_world->bodies, s_r_counts[rank], mpiBodyType, world->bodies, s_r_counts, displacements, mpiBodyType, ROOT,MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(l_world->bodies, s_r_counts[rank], mpiBodyType, NULL, s_r_counts, displacements, mpiBodyType, ROOT,MPI_COMM_WORLD);
        }
    }
    
    if(rank==ROOT){
        fprintf(stderr, "N-body took %10.3f seconds\n", end_t - start_t);

        print(world);

        filemap_close(&image_map);
    }

    free(world);
    free(l_world);
    free(s_r_counts);
    free(displacements);
    free(r_neighbours);
    free(g_forces_y);
    free(g_forces_x);

    MPI_Finalize();
    return 0;
}
