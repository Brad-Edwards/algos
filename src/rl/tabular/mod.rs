mod double_q;
mod monte_carlo;
mod q_learning;
mod sarsa;

pub use double_q::DoubleQLearning;
pub use monte_carlo::MonteCarloES;
pub use q_learning::QLearning;
pub use sarsa::Sarsa;
