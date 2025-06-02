import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime

print("=" * 80)
print("TUGAS: PARTICLE SWARM OPTIMIZATION (PSO)")
print("Implementasi Lengkap dengan Visualisasi")
print("Fungsi: f(x) = x²")
print("=" * 80)

class PSOOptimizer:
    def __init__(self, w=0.5, c1=1.5, c2=1.5, n_particles=10, max_iter=50, x_min=-10, x_max=10):
        """
        Inisialisasi PSO Optimizer
        
        Parameters:
        - w: Inertia weight (0.5)
        - c1: Cognitive coefficient (1.5)
        - c2: Social coefficient (1.5)
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
        self.iteration_calculations = []
        
    def objective_function(self, x):
        """Fungsi objektif: f(x) = x²"""
        return x ** 2
    
    def initialize_particles(self):
        """Inisialisasi partikel dengan posisi dan kecepatan acak"""
        print(f"\n1. INISIALISASI {self.n_particles} PARTIKEL:")
        print("-" * 50)
        
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
            
            print(f"Partikel {i}: pos={position:.4f}, vel={velocity:.4f}, f(x)={fitness:.4f}")
            
            # Update gBest
            if fitness < self.gBest_value:
                self.gBest_value = fitness
                self.gBest_position = position
        
        print(f"\nInisialisasi gBest: posisi={self.gBest_position:.4f}, nilai={self.gBest_value:.4f}")
        
        # Simpan ke history
        self.history.append({
            'iteration': 0,
            'gBest_value': self.gBest_value,
            'gBest_position': self.gBest_position,
            'particles': [p.copy() for p in self.particles]
        })
    
    def update_particles(self, iteration):
        """Update posisi dan kecepatan partikel untuk satu iterasi"""
        calculations = []
        calculations.append(f"\n=== ITERASI {iteration} ===")
        
        for i, particle in enumerate(self.particles):
            # Generate random numbers
            r1 = random.random()
            r2 = random.random()
            
            # Update velocity
            cognitive_component = self.c1 * r1 * (particle['pBest_position'] - particle['position'])
            social_component = self.c2 * r2 * (self.gBest_position - particle['position'])
            new_velocity = self.w * particle['velocity'] + cognitive_component + social_component
            
            calculations.append(
                f"Partikel {i}: v = {self.w} × {particle['velocity']:.3f} + "
                f"{self.c1} × {r1:.3f} × ({particle['pBest_position']:.3f} - {particle['position']:.3f}) + "
                f"{self.c2} × {r2:.3f} × ({self.gBest_position:.3f} - {particle['position']:.3f}) = {new_velocity:.3f}"
            )
            
            # Update position
            new_position = particle['position'] + new_velocity
            # Boundary check
            new_position = max(min(new_position, self.x_max), self.x_min)
            
            # Calculate new fitness
            new_fitness = self.objective_function(new_position)
            
            calculations.append(
                f"Partikel {i}: x = {particle['position']:.3f} + {new_velocity:.3f} = "
                f"{new_position:.3f}, f(x) = {new_fitness:.4f}"
            )
            
            # Update particle
            particle['position'] = new_position
            particle['velocity'] = new_velocity
            particle['current_value'] = new_fitness
            
            # Update pBest
            if new_fitness < particle['pBest_value']:
                particle['pBest_position'] = new_position
                particle['pBest_value'] = new_fitness
                calculations.append(f"Partikel {i}: pBest diperbarui ke {new_fitness:.4f}")
                
                # Update gBest
                if new_fitness < self.gBest_value:
                    self.gBest_position = new_position
                    self.gBest_value = new_fitness
                    calculations.append(
                        f"*** gBest diperbarui ke {self.gBest_value:.4f} pada posisi {self.gBest_position:.4f} ***"
                    )
        
        # Simpan ke history
        self.history.append({
            'iteration': iteration,
            'gBest_value': self.gBest_value,
            'gBest_position': self.gBest_position,
            'particles': [p.copy() for p in self.particles]
        })
        
        self.iteration_calculations.append(calculations)
        
        # Print progress setiap 10 iterasi atau 5 iterasi pertama
        if iteration <= 5 or iteration % 10 == 0:
            print(f"Iterasi {iteration}: gBest = {self.gBest_value:.6f} pada posisi {self.gBest_position:.4f}")
    
    def optimize(self):
        """Jalankan algoritma PSO"""
        print(f"\n2. PARAMETER PSO:")
        print(f"- Inersia (w): {self.w}")
        print(f"- Koefisien kognitif (c1): {self.c1}")
        print(f"- Koefisien sosial (c2): {self.c2}")
        print(f"- Jumlah partikel: {self.n_particles}")
        print(f"- Iterasi maksimum: {self.max_iter}")
        print(f"- Batas pencarian: [{self.x_min}, {self.x_max}]")
        
        # Inisialisasi partikel
        self.initialize_particles()
        
        print(f"\n3. PROSES ITERASI PSO:")
        print("=" * 60)
        
        # Iterasi PSO
        for iteration in range(1, self.max_iter + 1):
            self.update_particles(iteration)
        
        print(f"\n4. HASIL AKHIR:")
        print("=" * 60)
        print(f"Nilai minimum yang ditemukan: {self.gBest_value:.8f}")
        print(f"Posisi x terbaik: {self.gBest_position:.6f}")
        print(f"Jumlah iterasi: {self.max_iter}")
        
        # Konvergensi
        convergence_iteration = 0
        min_value = min([h['gBest_value'] for h in self.history])
        for i, h in enumerate(self.history):
            if h['gBest_value'] == min_value:
                convergence_iteration = i
                break
        print(f"Konvergensi tercapai pada iterasi: {convergence_iteration}")
        
        return self.gBest_position, self.gBest_value
    
    def print_detailed_calculations(self, max_iterations=3):
        """Cetak perhitungan detail untuk beberapa iterasi pertama"""
        print(f"\n5. PERHITUNGAN DETAIL (3 iterasi pertama):")
        print("=" * 80)
        
        for i, calculations in enumerate(self.iteration_calculations[:max_iterations]):
            for calc in calculations:
                print(calc)
    
    def print_particle_status(self):
        """Cetak status akhir semua partikel"""
        print(f"\n6. STATUS AKHIR PARTIKEL:")
        print("-" * 80)
        print(f"{'ID':<3} {'Posisi':<10} {'Kecepatan':<12} {'f(x)':<10} {'pBest Pos':<12} {'pBest Val':<12}")
        print("-" * 80)
        
        for particle in self.particles:
            print(f"{particle['id']:<3} {particle['position']:<10.4f} {particle['velocity']:<12.4f} "
                  f"{particle['current_value']:<10.4f} {particle['pBest_position']:<12.4f} "
                  f"{particle['pBest_value']:<12.6f}")
    
    def print_statistics(self):
        """Cetak statistik tambahan"""
        positions = [p['position'] for p in self.particles]
        initial_value = self.history[0]['gBest_value']
        
        print(f"\n7. STATISTIK TAMBAHAN:")
        print("-" * 40)
        print(f"- Rata-rata posisi akhir partikel: {np.mean(positions):.4f}")
        print(f"- Standar deviasi posisi: {np.std(positions):.4f}")
        print(f"- Jarak rata-rata dari optimum: {np.mean([abs(pos) for pos in positions]):.4f}")
        print(f"- Improvement dari iterasi awal: {(initial_value - self.gBest_value)/initial_value*100:.2f}%")
        
        print(f"\n8. VERIFIKASI:")
        print("-" * 40)
        print(f"- Solusi teoritis: x = 0, f(0) = 0")
        print(f"- Solusi PSO: x = {self.gBest_position:.6f}, f(x) = {self.gBest_value:.8f}")
        print(f"- Error absolut: {abs(self.gBest_value - 0):.8f}")
        print(f"- Error relatif posisi: {abs(self.gBest_position - 0):.6f}")
    
    def create_visualizations(self):
        """Buat visualisasi grafik"""
        print(f"\n9. MEMBUAT GRAFIK VISUALISASI...")
        print("-" * 40)
        
        # Prepare data
        iterations = [h['iteration'] for h in self.history]
        gBest_values = [h['gBest_value'] for h in self.history]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Convergence Plot (Log Scale)
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(iterations, gBest_values, 'b-o', linewidth=2, markersize=4)
        plt.title('Konvergensi Nilai Terbaik Global (gBest)', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Nilai f(x) Terbaik (log scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Annotate minimum
        min_idx = np.argmin(gBest_values)
        plt.annotate(f'Min: {min(gBest_values):.6f}\nIter: {min_idx}', 
                    xy=(min_idx, min(gBest_values)), 
                    xytext=(min_idx + 10, min(gBest_values) * 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')
        
        # 2. Function and Particles
        ax2 = plt.subplot(2, 3, 2)
        x_range = np.linspace(self.x_min, self.x_max, 1000)
        y_range = x_range ** 2
        
        plt.plot(x_range, y_range, 'g-', linewidth=2, label='f(x) = x²')
        plt.axvline(x=self.gBest_position, color='red', linestyle='--', linewidth=2, 
                   label=f'gBest = {self.gBest_position:.4f}')
        plt.axhline(y=self.gBest_value, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        # Plot final particle positions
        final_particles = self.history[-1]['particles']
        particle_positions = [p['position'] for p in final_particles]
        particle_values = [p['current_value'] for p in final_particles]
        plt.scatter(particle_positions, particle_values, c='blue', s=50, alpha=0.7, 
                   label='Posisi Akhir Partikel')
        
        # Plot pBest positions
        pBest_positions = [p['pBest_position'] for p in final_particles]
        pBest_values = [p['pBest_value'] for p in final_particles]
        plt.scatter(pBest_positions, pBest_values, c='orange', s=30, alpha=0.8, 
                   label='pBest Partikel')
        
        plt.title('Fungsi Objektif f(x) = x² dan Posisi Partikel', fontsize=12, fontweight='bold')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(self.x_min, self.x_max)
        
        # 3. Particle Movement Over Time
        ax3 = plt.subplot(2, 3, 3)
        for i in range(min(5, self.n_particles)):  # Show first 5 particles
            positions = [h['particles'][i]['position'] for h in self.history]
            plt.plot(iterations, positions, '-o', markersize=3, label=f'Partikel {i}')
        
        plt.axhline(y=self.gBest_position, color='red', linestyle='--', alpha=0.7, 
                   label='gBest Final')
        plt.title('Pergerakan Posisi Partikel', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Posisi x')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Velocity Over Time
        ax4 = plt.subplot(2, 3, 4)
        for i in range(min(5, self.n_particles)):  # Show first 5 particles
            velocities = [h['particles'][i]['velocity'] for h in self.history]
            plt.plot(iterations, velocities, '-o', markersize=3, label=f'Partikel {i}')
        
        plt.title('Pergerakan Kecepatan Partikel', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Kecepatan')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Diversity Over Time
        ax5 = plt.subplot(2, 3, 5)
        diversity = []
        for h in self.history:
            positions = [p['position'] for p in h['particles']]
            diversity.append(np.std(positions))
        
        plt.plot(iterations, diversity, 'purple', linewidth=2, marker='o', markersize=4)
        plt.title('Diversitas Swarm (Std Dev Posisi)', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Standar Deviasi')
        plt.grid(True, alpha=0.3)
        
        # 6. Improvement Rate
        ax6 = plt.subplot(2, 3, 6)
        improvements = []
        for i in range(1, len(gBest_values)):
            if gBest_values[i-1] != 0:
                improvement = (gBest_values[i-1] - gBest_values[i]) / gBest_values[i-1] * 100
                improvements.append(max(0, improvement))  # Only positive improvements
            else:
                improvements.append(0)
        
        plt.plot(iterations[1:], improvements, 'orange', linewidth=2, marker='o', markersize=4)
        plt.title('Rate Perbaikan per Iterasi (%)', fontsize=12, fontweight='bold')
        plt.xlabel('Iterasi')
        plt.ylabel('Perbaikan (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create summary statistics plot
        self.create_summary_plot()
    
    def create_summary_plot(self):
        """Buat plot ringkasan statistik"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Final particle distribution
        final_particles = self.history[-1]['particles']
        positions = [p['position'] for p in final_particles]
        values = [p['current_value'] for p in final_particles]
        
        ax1.hist(positions, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=self.gBest_position, color='red', linestyle='--', linewidth=2, 
                   label=f'gBest = {self.gBest_position:.4f}')
        ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Optimum = 0')
        ax1.set_title('Distribusi Posisi Akhir Partikel')
        ax1.set_xlabel('Posisi x')
        ax1.set_ylabel('Frekuensi')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Convergence comparison
        iterations = [h['iteration'] for h in self.history]
        gBest_values = [h['gBest_value'] for h in self.history]
        
        ax2.semilogy(iterations, gBest_values, 'b-', linewidth=2, label='PSO')
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Optimum Global')
        ax2.set_title('Konvergensi PSO (Skala Log)')
        ax2.set_xlabel('Iterasi')
        ax2.set_ylabel('Nilai Objektif (log)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter analysis
        parameters = ['w', 'c1', 'c2', 'n_particles']
        values = [self.w, self.c1, self.c2, self.n_particles]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = ax3.bar(parameters, values, color=colors, alpha=0.7)
        ax3.set_title('Parameter PSO yang Digunakan')
        ax3.set_ylabel('Nilai Parameter')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value}', ha='center', va='bottom')
        
        # 4. Performance metrics
        metrics = ['Error Absolut', 'Error Posisi', 'Std Dev Posisi', 'Improvement (%)']
        positions = [p['position'] for p in final_particles]
        initial_value = self.history[0]['gBest_value']
        
        metric_values = [
            abs(self.gBest_value),
            abs(self.gBest_position),
            np.std(positions),
            (initial_value - self.gBest_value) / initial_value * 100
        ]
        
        bars = ax4.bar(metrics, metric_values, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
        ax4.set_title('Metrik Performa PSO')
        ax4.set_ylabel('Nilai')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def save_results_to_csv(self):
        """Simpan hasil ke file CSV"""
        print(f"\n10. MENYIMPAN HASIL KE CSV...")
        print("-" * 40)
        
        # Prepare data for CSV
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
        
        # Summary statistics
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
    
    # Tampilkan pseudocode
    print("\nPSEUDOCODE ALGORITMA PSO:")
    print("=" * 50)
    pseudocode = """
    ALGORITMA PSO:
    1. Inisialisasi:
       - Buat n partikel dengan posisi dan kecepatan acak
       - Set pBest setiap partikel = posisi awal
       - Set gBest = pBest terbaik dari semua partikel

    2. Untuk setiap iterasi (1 sampai max_iter):
       a. Untuk setiap partikel i:
          - Hitung fitness f(xi)
          - Update pBest jika f(xi) < f(pBesti)
          - Update gBest jika f(pBesti) < f(gBest)
       
       b. Untuk setiap partikel i:
          - Update kecepatan: vi = w*vi + c1*r1*(pBesti - xi) + c2*r2*(gBest - xi)
          - Update posisi: xi = xi + vi
          - Batasi posisi dalam range yang diizinkan

    3. Return gBest dan f(gBest)
    """
    print(pseudocode)
    
    # Inisialisasi dan jalankan PSO
    pso = PSOOptimizer(
        w=0.5,      # Inertia weight
        c1=1.5,     # Cognitive coefficient  
        c2=1.5,     # Social coefficient
        n_particles=10,  # Jumlah partikel
        max_iter=50,     # Iterasi maksimum
        x_min=-10,       # Batas minimum
        x_max=10         # Batas maksimum
    )
    
    # Optimasi
    best_position, best_value = pso.optimize()
    
    # Tampilkan hasil detail
    pso.print_detailed_calculations()
    pso.print_particle_status()
    pso.print_statistics()
    
    # Buat visualisasi
    pso.create_visualizations()
    
    # Simpan hasil
    pso.save_results_to_csv()
    
    print(f"\n{'='*80}")
    print("TUGAS PSO SELESAI!")
    print(f"Nilai minimum ditemukan: {best_value:.8f}")
    print(f"Pada posisi x: {best_position:.6f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()