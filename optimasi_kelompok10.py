import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 80)
print("TUGAS: PARTICLE SWARM OPTIMIZATION (PSO)")
print("Implementasi Lengkap dengan Visualisasi")
print("Fungsi: f(x) = x²")
print("=" * 80)

# Tetapkan seed untuk hasil yang konsisten
random.seed(42)

class PSOOptimizer:
    def __init__(self, w=0.8, c1=1.0, c2=1.0, n_particles=10, max_iter=50, x_min=-10, x_max=10):
        """
        Inisialisasi PSO Optimizer
        
        Parameters:
        - w: Inertia weight (0.8)
        - c1: Cognitive coefficient (1.0)
        - c2: Social coefficient (1.0)
        - n_particles: Jumlah partikel (10)
        - max_iter: Iterasi maksimum (50)
        - x_min, x_max: Batas pencarian (-10, 10)
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        
        # Inisialisasi variabel
        self.particles = []
        self.gBest_position = 0
        self.gBest_value = float('inf')
        self.history = []
        
    def objective_function(self, x):
        """Fungsi objektif: f(x) = x²"""
        return x ** 2
    
    def initialize_particles(self):
        """Inisialisasi partikel dengan posisi dan kecepatan acak"""
        self.particles = []
        for i in range(self.n_particles):
            position = random.uniform(self.x_min, self.x_max)
            velocity = random.uniform(-(abs(self.x_max - self.x_min) * 0.1), 
                                    (abs(self.x_max - self.x_min) * 0.1))
            fitness = self.objective_function(position)
            
            particle = {
                'id': i,
                'position': position,
                'velocity': velocity,
                'pBest_position': position,
                'pBest_value': fitness,
                'current_value': fitness
            }
            self.particles.append(particle)
            
            if fitness < self.gBest_value:
                self.gBest_value = fitness
                self.gBest_position = position
        
        self.history.append({
            'iteration': 0,
            'gBest_value': self.gBest_value,
            'gBest_position': self.gBest_position,
            'particles': [p.copy() for p in self.particles]
        })
    
    def update_particles(self, iteration):
        """Update posisi dan kecepatan partikel untuk satu iterasi"""
        for i, particle in enumerate(self.particles):
            r1 = random.random()
            r2 = random.random()
            
            cognitive_component = self.c1 * r1 * (particle['pBest_position'] - particle['position'])
            social_component = self.c2 * r2 * (self.gBest_position - particle['position'])
            new_velocity = self.w * particle['velocity'] + cognitive_component + social_component
            
            new_position = particle['position'] + new_velocity
            new_position = max(min(new_position, self.x_max), self.x_min)
            
            new_fitness = self.objective_function(new_position)
            
            particle['position'] = new_position
            particle['velocity'] = new_velocity
            particle['current_value'] = new_fitness
            
            if new_fitness < particle['pBest_value']:
                particle['pBest_position'] = new_position
                particle['pBest_value'] = new_fitness
                
                if new_fitness < self.gBest_value:
                    self.gBest_position = new_position
                    self.gBest_value = new_fitness
        
        self.history.append({
            'iteration': iteration,
            'gBest_value': self.gBest_value,
            'gBest_position': self.gBest_position,
            'particles': [p.copy() for p in self.particles]
        })
    
    def optimize(self):
        """Jalankan algoritma PSO"""
        self.initialize_particles()
        
        for iteration in range(1, self.max_iter + 1):
            self.update_particles(iteration)
        
        # Cari nilai minimum absolut dari seluruh riwayat
        gBest_values = [h['gBest_value'] for h in self.history]
        min_idx = np.argmin(gBest_values)
        self.gBest_value = gBest_values[min_idx]
        self.gBest_position = self.history[min_idx]['gBest_position']
        
        return self.gBest_position, self.gBest_value
    
    def create_visualizations(self):
        """Buat visualisasi grafik nilai terbaik per iterasi"""
        print("\nMEMBUAT GRAFIK VISUALISASI...")
        print("-" * 40)
        
        iterations = [h['iteration'] for h in self.history]
        gBest_values = [h['gBest_value'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, gBest_values, 'b-o', linewidth=2, markersize=4, label='Nilai Terbaik (gBest)')
        plt.title('Nilai Terbaik Global (gBest) per Iterasi', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Nilai f(x) Terbaik')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.show()
    
    def save_results_to_csv(self):
        """Simpan hasil ke file CSV"""
        print("\nMENYIMPAN HASIL KE CSV...")
        print("-" * 40)
        
        data = []
        for h in self.history:
            for p in h['particles']:
                data.append({
                    'iteration': h['iteration'],
                    'particle_id': p['id'],
                    'position': p['position'],
                    'velocity': p['velocity'],
                    'current_value': p['current_value'],
                    'pBest_position': p['pBest_position'],
                    'pBest_value': p['pBest_value'],
                    'gBest_position': h['gBest_position'],
                    'gBest_value': h['gBest_value']
                })
        
        df = pd.DataFrame(data)
        filename = f"pso_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Hasil disimpan ke: {filename}")
        
        summary_data = {
            'Parameter': ['w', 'c1', 'c2', 'n_particles', 'max_iter', 'x_min', 'x_max'],
            'Value': [self.w, self.c1, self.c2, self.n_particles, self.max_iter, self.x_min, self.x_max]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"pso_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Parameter disimpan ke: {summary_filename}")

# MAIN EXECUTION
def main():
    """Fungsi utama untuk menjalankan PSO"""
    pso = PSOOptimizer(
        w=0.8,
        c1=1.0,
        c2=1.0,
        n_particles=10,
        max_iter=50,
        x_min=-10,
        x_max=10
    )
    
    best_position, best_value = pso.optimize()
    
    print("\nHASIL AKHIR:")
    print("=" * 60)
    print(f"Nilai minimum yang ditemukan: {best_value:.20f}")
    print(f"Posisi x terbaik: {best_position:.20f}")
    
    pso.create_visualizations()
    pso.save_results_to_csv()
    
    print(f"\n{'='*80}")
    print("TUGAS PSO SELESAI!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()