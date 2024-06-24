from tkinter import ttk
import tkinter as tk
import test_functions
from animate import animate

def get_data_and_run():
    generations = int(max_gens_ent.get())
    population_size = int(pop_size_ent.get())
    selected_algorithm = test_function_drp.get()
    func = getattr(test_functions, selected_algorithm.lower())
    bounds = getattr(test_functions.Bounds, selected_algorithm).value
    print(f'calling: {selected_algorithm}')
    animate(generations,population_size, test_function=func, test_function_bounds=bounds)

def remember_selected(selection):
    selected_algorithm = selection
    print(selected_algorithm)



window = tk.Tk()
window.geometry('200x200')

#Frames
max_gens_frm = tk.Frame(master=window).pack(padx=5, pady=5)
pop_size_frm = tk.Frame(master=window).pack()
test_function_frm = tk.Frame(master=window).pack()

#Labels
max_gens_lbl = tk.Label(master=max_gens_frm, text='Numarul maxim de generatii')
pop_size_lbl = tk.Label(master=pop_size_frm, text='Dimensiunea populatiei')
test_function_lbl = tk.Label(master=test_function_frm, text='Functia de test')

# Entries
max_gens_ent = tk.Entry(master=max_gens_frm)
pop_size_ent = tk.Entry(master=pop_size_frm)

# Dropdown
options = test_functions.Bounds._member_names_
test_function_drp = ttk.Combobox(master=test_function_frm, state='readonly', values=options)
test_function_drp.current(0)

run_btn = tk.Button(master=window, text="Ruleaza", command=get_data_and_run)

# Pack elements
max_gens_lbl.pack()
max_gens_ent.pack(pady=5)

pop_size_lbl.pack()
pop_size_ent.pack(pady=5)

test_function_lbl.pack()
test_function_drp.pack()
run_btn.pack(pady=5)

window.mainloop()

