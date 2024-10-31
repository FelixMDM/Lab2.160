// A simple 5-point stencil
//
// Think of this as a hot plate with heaters/coolers that keep the
// 4 edges (b1,b3,b4,b6) and 4 corners (b0,b2,b5,b7) at a constant temp.
//
//     B0 B1B1B1 B2
//     B3 I I I  B4
//     B3 I I I  B4
//     B3 I I I  B4
//     B5 B6B6B6 B7

// At every time step, we recompute the new temperature of our cell
// by averaging our temperature with the temps of the cells around us.
//
// This evolves the plate into some pretty patterns
//
// To run locally, we build this
// % c++ hotplate.cpp -o hotplate -pthread
//
// ./hotplate N B0 B1 B2 B3 B4 B5 B6 B7 steps output.dat [tasks]
//
// To run a simple problem (100x100 with a hot bottom edge
// where hot is 100, the inner temp starts at 1 and the other
// boundaries are held at temp 1
// ./hotplate 100   1 1 1 1 1 100 100 100 1  1000 foo.dat
// the first 100 is a size, the next 8 values are boundaries.
// the next 1 is the inner temp.  The 1000 is the step count
// foo.dat will receive the 10,000 double precision values
// The task count will default to 1 (unless you are running on
// the cluster).
//
// The output is a bunch of double precision values.  I've provided
// a python script to convert them into a pretty .png file
//
// python3 heatmap.py --data=output.dat --png=pretty.png

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

class Task {
  int id;
  int ntasks;
  double* to;
  double* from;
  int N;
  int steps;
public:
  static void* runner(void* arg) {
    reinterpret_cast<Task*>(arg)->execute();
    return nullptr;
  }

  Task(int id, int ntasks, double* to, double* from, int N, int steps)
    : id(id), ntasks(ntasks), to(to), from(from), N(N), steps(steps)
  {
  }

  void execute() {
    for(int i=0; i<steps;++i) {
      std::swap(to,from);

      // For the inner core, we apply a stencil and make the new temperature
      // the average of our current temp and those cells around us.  We'll
      // think of this as taking the average of the N / S / E / W of us...
      // this is often called a NEWS average
      for(int x=1; x<N-1; ++x) {
	for(int y=1; y<N-1; ++y) {
	  // This is my cell
	  double cell = from[x*N + y];
	
	  // north is previous row, same column
	  double north = from[(x-1)*N + y];

	  // south is next row, same column
	  double south = from[(x+1)*N + y];

	  // east is from thex right of current
	  double east = from[x*N + y+1];

	  // west is from the left
	  double west = from[x*N + y-1];

	  // We update "from" with the new, averaged value
	  to[x*N + y] = (cell+north+south+east+west)/5.0;
	}
      }
    };

  }
};

// ./hotplate N b0 b1 b2 b3 b4 b5 b6 b7 inner steps output optional_ntasks
int main(int argc, char* argv[]) {
  if (argc != 13 and argc != 14) {
    std::cerr << "Usage: " << argv[0] << ' '
	      << "N B0 B1 B2 B3 B4 B5 B6 B7 steps output.dat [tasks]\n";
    std::cerr << "N      : create a NxN plate\n";
    std::cerr << "B0-B7  : the 8 boundary temperatures\n";
    std::cerr << "inner  : the temperature everywhere else\n";
    std::cerr << "steps  : the number of iterations to evolve\n";
    std::cerr << "output : the filename that receives the raw output\n";
    std::cerr << "ntasks : [optional] how many tasks to run\n";
    std::cerr << "\n";
    std::cerr << "ntasks defaults to 1 unless you are using srun -nX\n";
    std::cerr << "on the cluster in which case it defaults to X\n";
    return 1;
  }

  // We assume this to be a square
  int N = std::stoi(argv[1]);

  //  b0           b1             b2
  //
  //  b3                          b4
  //
  //  b5           b6             b7
  double B0 = std::stod(argv[2]);
  double B1 = std::stod(argv[3]);
  double B2 = std::stod(argv[4]);
  double B3 = std::stod(argv[5]);
  double B4 = std::stod(argv[6]);
  double B5 = std::stod(argv[7]);
  double B6 = std::stod(argv[8]);
  double B7 = std::stod(argv[9]);

  // inner temperature everywhere else
  double inner = std::stod(argv[10]);

  // Number of steps
  int steps = std::stoi(argv[11]);
  
  // Where to dump the output
  std::string output_filename(argv[12]);

  // How many threads to use?
  int ntasks = 1;
  const char* env_tasks = ::getenv("SLURM_STEP_NUM_TASKS");
  if (env_tasks) {
    ntasks = std::stoi(env_tasks);
  }
  if (argc == 14) ntasks = std::stoi(argv[13]);

  // The way this often works is that we have a "from" data set and
  // a "to" data set that we then swap at the end of the step
  double* from = new double[N*N];
  double* to = new double[N*N];

  // I'll set the "to" and "from" boundary conditions and the
  // inner core values while I'm at it (from will be overwritten)
  for(int i=0;i<N*N;++i) {
    to[i] = inner;
    from[i] = inner;
  }

  to[0*N+0] = B0;
  from[0*N+0] = B0;

  for(int i=1; i<N-1; ++i) {
    to[0*N+i] = B1;
    from[0*N+i] = B1;
  }
  
  to[0*N+N-1] = B2;
  from[0*N+N-1] = B2;

  for(int i=1; i<N-1; ++i) {
    to[i*N+0] = B3;
    from[i*N+0] = B3;
  }

  for(int i=1; i<N-1; ++i) {
    to[i*N+N-1] = B4;
    from[i*N+N-1] = B4;
  }
  
  to[N*(N-1)+0] = B5;
  from[N*(N-1)+0] = B5;

  for(int i=1; i<N-1; ++i) {
    to[N*(N-1)+i] = B6;
    from[N*(N-1)+i] = B6;
  }
  
  to[N*(N-1)+N-1] = B7;
  from[N*(N-1)+N-1] = B7;

  // Build a vector of tasks to do
  std::vector<Task> tasks;
  tasks.emplace_back(0,1, to,from,N,steps);

  // Start a timer
  high_resolution_clock::time_point begin = high_resolution_clock::now();
  
  // Define the threads we want to run and get them started
  std::vector<pthread_t> threads(tasks.size());
  if (ntasks > tasks.size()) ntasks = tasks.size();

  for(int i=0; i < ntasks; ++i) {
    int status = ::pthread_create(&threads[i], nullptr, Task::runner, &tasks[i]);
    if (status != 0) {
      ::perror("thread create");
      return 1;
    }
  }

  // Wait for all to finish
  for(int i=0; i < ntasks; ++i) {
    pthread_join(threads[i],nullptr);
  }

  auto time_span = duration_cast<duration<double>>(high_resolution_clock::now() - begin);

  std::cerr << ntasks << ' ' << time_span.count() << '\n';

  // Dump the data in binary form (local endian) to the output file
  int fd = ::creat(output_filename.c_str(), 0644);
  if (fd < 0) {
    throw std::invalid_argument(output_filename);
  }
  for(int i=0; i<N*N; ++i) {
    ssize_t n = ::write(fd, to+i, sizeof(double));
    if (n != sizeof(double)) {
      ::perror("write");
      return 1;
    }
  }
  ::close(fd);

  // Good form to clean up allocated memory
  delete [] to;
  delete [] from;
  return 0;
}