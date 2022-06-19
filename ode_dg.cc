/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2021 by the deal.II authors
 *
 * This file is based on step-3 of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Jan Philipp Thiele, Julian Roth 2022
 *
 */

//Most of these should be known if you ran the Poisson example (step-3)
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

// This file contains the description of the Lagrange interpolation 
// discontinuous finite element:
#include <deal.II/fe/fe_dgq.h>


#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// This class is based on the main class of step-3, but here we want to model 
// the ODE y' = lambda * y.   
// The main differences lie in using discontinuous Galerkin (DG) elements, 
// i.e. FE_DGQ and in including the ODE parameter lambda.
class ODE
{
public:
  ODE(int degree = 1,double lambda = 5., double T = 1.);

  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<1> triangulation;
  FE_DGQ<1>          fe;
  DoFHandler<1>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
  double lambda;
  double T;
};

ODE::ODE(int degree, double lambda, double T)
  : fe(degree)
  , dof_handler(triangulation)
  , lambda(lambda)
  , T(T)
{}


void ODE::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0, T);
  triangulation.refine_global(2);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

void ODE::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
            
  // generate the dynamic sparsity pattern as usual
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  
  // to account for the jump terms we need to add the index to the sparsity pattern
  for (types::global_dof_index i = 1; i< dof_handler.n_dofs() ; i++ )
  {
    dsp.add(i,i-1); 
  }
  
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


void ODE::assemble_system()
{
  
  QGauss<1> quadrature_formula(fe.degree + 2);
  FEValues<1> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
                                 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);

  // As we have an ODE shape_grad does not coincide with the 
  // nabla Operator which usually only contains spatial derivatives
  // But as our single dimension is time the shape_grad contains the 
  // temporal derivative
  
  for (const auto &cell : dof_handler.active_cell_iterators())
  {  
      fe_values.reinit(cell);

      cell_matrix = 0;
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
            {
              
              // (y',phi)
              cell_matrix(i, j) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 fe_values.shape_grad(j, q_index)[0] * // dt phi_j(x_q)
                 fe_values.JxW(q_index));// dx
                
              // -(lambda*y,phi)
              cell_matrix(i,j) -=
                (lambda*
                 fe_values.shape_value(i,q_index) *
                 fe_values.shape_value(j,q_index) * 
                 fe_values.JxW(q_index));
            }
        }
      cell->get_dof_indices(local_dof_indices);
      
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));          
          
          
      // Additionally we need the jump terms [y]_{m-1}*phi^+(t_m)
      // which evaluate to 1*y_{m-1}^+ in the row of phi_m^+
      system_matrix.add(local_dof_indices[0],
                          local_dof_indices[0],
                          1.0);
      // and either
      if (cell->active_cell_index() == 0)
      {
        //y0 = 1 in the rhs for the initial edge at t=0
          system_rhs[0]=1.;
      }
      else {
        //or -1*y_{m-1}^- for all other element edges with t>0
        //apart from the final edge
        std::prev(cell)->get_dof_indices(neighbor_dof_indices);
        system_matrix.add(local_dof_indices[0],
                          neighbor_dof_indices[dofs_per_cell-1],
                          -1.0);
      }
    }
}

// As our Matrix is no longer symmetrical we cannot use the CG solver
// the easiest approach is using a direct solver which for a single dimension
// is still viable in terms of computational and memory costs
void ODE::solve()
{

 SparseDirectUMFPACK solver;
 solver.initialize(system_matrix);
 solver.vmult(solution, system_rhs);
}

// As the normal output basically interpolates the solution 
// back to a linear finite element, we cannot properly view
// the solution for a higher polynomial degree. 
// For PDEs in more dimension this is mostly fine, but for an ODE
// it obstructs the shape of the solution.
// For this reason the solution is evaluated at multiple inner points
// as well as on the boundary of each element.
// The timepoints and matching solutions are then written to the console
// in a matlab compatible format and can be plotted using plot(t,u)
void ODE::output_results() const
{
  unsigned int n_points_per_cell = 11;
  if ( fe.get_degree() < 2)
  {
    n_points_per_cell = 2; //for constant and linear only use 2 points.
  }
  
  //Iterated formula splits element into (n_points_per_cell-1) subintervals on which the trapezoidal rule is applied. 
  //This results in n_points_per_cell uniformly distributed quadrature points
  QIterated<1> quad(QTrapez<1>(), n_points_per_cell-1); // need to use QTrapezoid for deal.II 9.3.0
  FEValues<1> fe_values(fe, quad, update_values | update_quadrature_points);
  std::vector< std::vector< Point<1> > > q_points(triangulation.n_active_cells(), std::vector< Point<1> >(n_points_per_cell, Point<1>(0.)));
  std::vector< std::vector<double> > u_values(triangulation.n_active_cells(), std::vector<double>(n_points_per_cell));
  
  //Get all "quadrature"-points and corresponding solution values.
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    fe_values.get_function_values(solution,u_values[cell->index()]);
    q_points[cell->index()] = fe_values.get_quadrature_points();
  }
  
  //Output points as a Matlab matrix
  std::cout << "t = [";
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    unsigned int i = 0;
    for ( ; i < n_points_per_cell-1 ; i++ )
    {
      std::cout << q_points[cell->index()][i] << ", ";  
    }
    std::cout << q_points[cell->index()][i];
    if (std::next(cell) != dof_handler.end()) {
      std::cout << "; "; 
    }
  }
  std::cout << "];" << std::endl;
  
  //Output corresponding solutions as a Matlab matrix
  std::cout << "u = [";
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    unsigned int i = 0;
    for ( ; i < n_points_per_cell-1 ; i++ )
    {
      std::cout << u_values[cell->index()][i] << ", ";  
    }
    std::cout << u_values[cell->index()][i];
    if (std::next(cell) != dof_handler.end()) {
      std::cout << "; "; 
    }
  }
  std::cout << "];" << std::endl;
 
}


void ODE::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main()
{
  deallog.depth_console(2);
  int fe_degree = 1;
  double lambda = 5.0;
  double T = 1.0;
  ODE laplace_problem(fe_degree,lambda,T);
  laplace_problem.run();

  return 0;
}
