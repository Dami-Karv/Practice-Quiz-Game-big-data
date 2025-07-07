import tkinter as tk
from tkinter import ttk
import random
import math
from questions import QUIZ_QUESTIONS


class CircularProgress:
    def __init__(self, parent, size=80):
        self.size = size
        self.canvas = tk.Canvas(parent, width=size, height=size, bg='#0a0a0f', highlightthickness=0)
        self.canvas.pack()

        self.bg_color = '#1e293b'
        self.correct_color = '#10b981'
        self.wrong_color = '#ef4444'

        self.total_questions = 0
        self.correct_answers = 0
        self.wrong_answers = 0

    def update_progress(self, total, correct, wrong):
        self.total_questions = total
        self.correct_answers = correct
        self.wrong_answers = wrong

        self.canvas.delete("all")

        if total > 0:
            correct_angle = (correct / total) * 360
            wrong_angle = (wrong / total) * 360
        else:
            correct_angle = wrong_angle = 0

        center = self.size // 2
        radius = (self.size // 2) - 10

        self.canvas.create_oval(center - radius, center - radius,
                                center + radius, center + radius,
                                outline=self.bg_color, width=6, fill="")

        if correct_angle > 0:
            self.canvas.create_arc(center - radius, center - radius,
                                   center + radius, center + radius,
                                   start=90, extent=-correct_angle,
                                   outline=self.correct_color, width=6,
                                   style="arc")

        if wrong_angle > 0:
            self.canvas.create_arc(center - radius, center - radius,
                                   center + radius, center + radius,
                                   start=90 - correct_angle, extent=-wrong_angle,
                                   outline=self.wrong_color, width=6,
                                   style="arc")

        if total > 0:
            percentage = round((correct / total) * 100)
            self.canvas.create_text(center, center - 5,
                                    text=f"{percentage}%",
                                    font=('Segoe UI', 12, 'bold'),
                                    fill='#f8fafc')

            self.canvas.create_text(center, center + 10,
                                    text=f"{correct}/{total}",
                                    font=('Segoe UI', 8, 'normal'),
                                    fill='#94a3b8')


class UltraModernQuizApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("485 Quiz Gia Alania")
        self.root.geometry("1100x750")
        self.root.configure(bg='#0a0a0f')

        self.colors = {
            'bg': '#0a0a0f',
            'card_bg': '#1a1a2e',
            'accent': '#16213e',
            'primary': '#6366f1',
            'secondary': '#8b5cf6',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'text': '#f8fafc',
            'text_secondary': '#94a3b8',
            'text_muted': '#64748b',
            'surface': '#0f172a',
            'border': '#1e293b'
        }

        self.questions = QUIZ_QUESTIONS
        self.current_questions = []
        self.current_question_index = 0
        self.score = 0
        self.user_answers = []
        self.selected_answer = tk.StringVar()
        self.quiz_name = ""
        self.feedback_visible = False
        self.current_shuffled_options = {}
        self.current_correct_letter = None

        self.include_mc = tk.BooleanVar(value=True)
        self.include_input = tk.BooleanVar(value=True)
        self.include_tf = tk.BooleanVar(value=True)
        self.randomize_questions = tk.BooleanVar(value=False)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = None

        self.setup_ui()
        self.center_window()
        self.show_main_menu()

        self.timer_start = None
        self.timer_running = False
        self.elapsed_time = 0
        self.timer_label = None
        self.timer_after_id = None
        self.current_quiz_randomized = False

        self.root.bind('<Configure>', self.on_window_resize)
        self.root.bind('<KeyPress>', self.handle_keypress)

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill='both', expand=True, padx=40, pady=20)

        self.canvas = tk.Canvas(self.main_container, bg=self.colors['bg'], highlightthickness=0, bd=0)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Vertical.TScrollbar",
                        background=self.colors['bg'],
                        troughcolor=self.colors['bg'],
                        bordercolor=self.colors['bg'],
                        arrowcolor=self.colors['text_muted'],
                        darkcolor=self.colors['surface'],
                        lightcolor=self.colors['surface'],
                        relief='flat',
                        borderwidth=0)

        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview,
                                       style="Custom.Vertical.TScrollbar")
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind('<Configure>', self.configure_canvas_width)

        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", on_mousewheel)

    def configure_canvas_width(self, event=None):
        if event:
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas.find_all()[0], width=canvas_width)

    def handle_keypress(self, event):
        if event.keysym == 'space':
            focused_widget = self.root.focus_get()
            if not isinstance(focused_widget, tk.Entry):
                self.submit_answer()
                return 'break'

    def on_window_resize(self, event=None):
        if event and event.widget == self.root:
            width = self.root.winfo_width()
            padding = max(20, min(80, (width - 900) // 4))
            self.main_container.configure(padx=padding)

    def clear_content(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def create_ultra_modern_card(self, parent, width=700, minimal=False):
        container = tk.Frame(parent, bg=self.colors['bg'])
        container.pack(fill='x', pady=6)

        if minimal:
            card = tk.Frame(container,
                            bg=self.colors['surface'],
                            relief='flat',
                            bd=0)
        else:
            card = tk.Frame(container,
                            bg=self.colors['card_bg'],
                            relief='flat',
                            bd=1,
                            highlightbackground=self.colors['border'],
                            highlightthickness=1)

        card.pack(anchor='center')
        return card

    def create_sleek_title(self, parent, text, size=26, color=None):
        container = tk.Frame(parent, bg=self.colors['bg'])
        container.pack(fill='x', pady=(0, 8))

        title = tk.Label(container,
                         text=text,
                         font=('SF Pro Display', size, 'normal'),
                         bg=self.colors['bg'],
                         fg=color or self.colors['text'])
        title.pack()
        return title

    def create_sleek_subtitle(self, parent, text, size=11, color=None):
        container = tk.Frame(parent, bg=self.colors['bg'])
        container.pack(fill='x', pady=(0, 25))

        subtitle = tk.Label(container,
                            text=text,
                            font=('SF Pro Display', size, 'normal'),
                            bg=self.colors['bg'],
                            fg=color or self.colors['text_secondary'],
                            wraplength=700)
        subtitle.pack()
        return subtitle

    def start_timer(self):
        import time
        self.timer_start = time.time()
        self.timer_running = True
        self.elapsed_time = 0
        self.update_timer_display()

    def stop_timer(self):
        if self.timer_running:
            import time
            self.elapsed_time = time.time() - self.timer_start
            self.timer_running = False
            if self.timer_after_id:
                self.root.after_cancel(self.timer_after_id)
        return self.elapsed_time

    def update_timer_display(self):
        if self.timer_running and self.timer_start:
            import time
            current_time = time.time() - self.timer_start
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)

            if self.timer_label:
                self.timer_label.config(text=f"⏱️ {minutes:02d}:{seconds:02d}")

            self.timer_after_id = self.root.after(1000, self.update_timer_display)

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def show_success_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Success")
        popup.geometry("300x100")
        popup.configure(bg=self.colors['card_bg'])
        popup.transient(self.root)
        popup.grab_set()

        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 350, self.root.winfo_rooty() + 250))

        tk.Label(popup, text=message, font=('Segoe UI', 11, 'normal'),
                 bg=self.colors['card_bg'], fg=self.colors['success']).pack(expand=True)

        tk.Button(popup, text="OK", command=popup.destroy,
                  bg=self.colors['success'], fg='white').pack(pady=10)

    def create_sleek_button(self, parent, text, command, style="primary", width=None):
        if style == "primary":
            bg_color = self.colors['primary']
            fg_color = '#ffffff'
            active_bg = '#7c3aed'
            hover_bg = '#7c3aed'
        elif style == "special":
            bg_color = self.colors['success']
            fg_color = '#ffffff'
            active_bg = '#059669'
            hover_bg = '#059669'
        elif style == "ghost":
            bg_color = self.colors['bg']
            fg_color = self.colors['text_secondary']
            active_bg = self.colors['surface']
            hover_bg = self.colors['surface']
        else:
            bg_color = self.colors['surface']
            fg_color = self.colors['text']
            active_bg = self.colors['accent']
            hover_bg = self.colors['accent']

        button = tk.Button(parent,
                           text=text,
                           command=command,
                           font=('SF Pro Display', 9, 'normal'),
                           bg=bg_color,
                           fg=fg_color,
                           activebackground=active_bg,
                           activeforeground=fg_color,
                           relief='flat',
                           bd=0,
                           padx=24,
                           pady=12,
                           cursor='hand2')

        if width:
            button.config(width=width)

        def on_enter(e):
            button.config(bg=hover_bg)

        def on_leave(e):
            button.config(bg=bg_color)

        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)

        return button

    def create_sleek_checkbox(self, parent, text, variable):
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill='x', pady=2)

        checkbox = tk.Checkbutton(container,
                                  text=text,
                                  variable=variable,
                                  font=('SF Pro Display', 10, 'normal'),
                                  bg=self.colors['card_bg'],
                                  fg=self.colors['text'],
                                  selectcolor=self.colors['primary'],
                                  activebackground=self.colors['accent'],
                                  activeforeground=self.colors['text'],
                                  bd=0,
                                  highlightthickness=0,
                                  padx=16,
                                  pady=8)
        checkbox.pack(anchor='w')

        def on_enter(e):
            checkbox.config(bg=self.colors['accent'])

        def on_leave(e):
            checkbox.config(bg=self.colors['card_bg'])

        checkbox.bind('<Enter>', on_enter)
        checkbox.bind('<Leave>', on_leave)

        return checkbox

    def create_sleek_radiobutton(self, parent, text, variable, value):
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill='x', pady=1)

        radio = tk.Radiobutton(container,
                               text=text,
                               variable=variable,
                               value=value,
                               font=('SF Pro Display', 11, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'],
                               selectcolor=self.colors['primary'],
                               activebackground=self.colors['accent'],
                               activeforeground=self.colors['text'],
                               indicatoron=1,
                               bd=0,
                               relief='flat',
                               padx=20,
                               pady=10)
        radio.pack(anchor='w', fill='x')

        def on_enter(e):
            radio.config(bg=self.colors['accent'])

        def on_leave(e):
            radio.config(bg=self.colors['card_bg'])

        radio.bind('<Enter>', on_enter)
        radio.bind('<Leave>', on_leave)

        self.add_copy_menu(radio, text)

        return radio

    def show_main_menu(self):
        self.clear_content()

        self.create_sleek_title(self.scrollable_frame, "485 Quiz", 32)
        self.create_sleek_subtitle(self.scrollable_frame, "Machine Learning Assessment Platform", 12)

        stats_card = self.create_ultra_modern_card(self.scrollable_frame, 550, minimal=True)
        stats_content = tk.Frame(stats_card, bg=self.colors['surface'])
        stats_content.pack(fill='x', padx=25, pady=16)

        total_questions = sum(len(questions) for questions in self.questions.values())

        stats_text = f"{len(self.questions)} Quizzes • {total_questions} Questions"
        stats_label = tk.Label(stats_content,
                               text=stats_text,
                               font=('SF Pro Display', 11, 'normal'),
                               bg=self.colors['surface'],
                               fg=self.colors['primary'])
        stats_label.pack()

        descriptions = {
            "Quiz #1": "NumPy • pandas • matplotlib • statistics",
            "Quiz #2": "PCA • neural networks • ML algorithms",
            "Quiz #3": "CNNs • RNNs • transformers • vision",
            "Quiz #4": "NLP • video analysis • reinforcement learning"
        }

        for quiz_name, questions in self.questions.items():
            card = self.create_ultra_modern_card(self.scrollable_frame, 650)
            content = tk.Frame(card, bg=self.colors['card_bg'])
            content.pack(fill='x', padx=28, pady=18)

            title_label = tk.Label(content,
                                   text=quiz_name,
                                   font=('SF Pro Display', 15, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text'],
                                   anchor='w')
            title_label.pack(fill='x', pady=(0, 4))

            if quiz_name in descriptions:
                desc_label = tk.Label(content,
                                      text=descriptions[quiz_name],
                                      font=('SF Pro Display', 9, 'normal'),
                                      bg=self.colors['card_bg'],
                                      fg=self.colors['text_secondary'],
                                      anchor='w')
                desc_label.pack(fill='x', pady=(0, 12))

            bottom_frame = tk.Frame(content, bg=self.colors['card_bg'])
            bottom_frame.pack(fill='x')

            count_label = tk.Label(bottom_frame,
                                   text=f"{len(questions)} questions",
                                   font=('SF Pro Display', 8, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text_muted'])
            count_label.pack(side='left')

            start_btn = self.create_sleek_button(bottom_frame,
                                                 "Configure →",
                                                 lambda name=quiz_name: self.show_quiz_setup(name),
                                                 "primary")
            start_btn.pack(side='right')

        everything_card = self.create_ultra_modern_card(self.scrollable_frame, 650)
        everything_content = tk.Frame(everything_card, bg=self.colors['card_bg'])
        everything_content.pack(fill='x', padx=28, pady=18)

        everything_title = tk.Label(everything_content,
                                    text="Everything Test",
                                    font=('SF Pro Display', 15, 'normal'),
                                    bg=self.colors['card_bg'],
                                    fg=self.colors['success'],
                                    anchor='w')
        everything_title.pack(fill='x', pady=(0, 4))

        everything_desc = tk.Label(everything_content,
                                   text="All questions • randomized • no repeats",
                                   font=('SF Pro Display', 9, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text_secondary'],
                                   anchor='w')
        everything_desc.pack(fill='x', pady=(0, 12))

        everything_bottom = tk.Frame(everything_content, bg=self.colors['card_bg'])
        everything_bottom.pack(fill='x')

        everything_count = tk.Label(everything_bottom,
                                    text=f"{total_questions} questions",
                                    font=('SF Pro Display', 8, 'normal'),
                                    bg=self.colors['card_bg'],
                                    fg=self.colors['text_muted'])
        everything_count.pack(side='left')

        everything_btn = self.create_sleek_button(everything_bottom,
                                                  "Configure →",
                                                  self.show_everything_setup,
                                                  "special")
        everything_btn.pack(side='right')

        footer_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        footer_container.pack(fill='x', pady=25)

        footer_frame = tk.Frame(footer_container, bg=self.colors['bg'])
        footer_frame.pack()

        self.create_sleek_button(footer_frame, "Random", self.show_random_setup, "ghost").pack(side='left', padx=3)
        self.create_sleek_button(footer_frame, "Exit", self.root.quit, "ghost").pack(side='left', padx=3)

    def show_everything_setup(self):
        self.clear_content()

        self.create_sleek_title(self.scrollable_frame, "Configure Everything Test", 22)
        self.create_sleek_subtitle(self.scrollable_frame, "Choose question types from all quizzes", 11)

        all_questions = []
        for questions in self.questions.values():
            all_questions.extend(questions)

        mc_count = len([q for q in all_questions if q.get('type', 'multiple_choice') == 'multiple_choice'])
        input_count = len([q for q in all_questions if q.get('type') == 'input'])
        tf_count = len([q for q in all_questions if q.get('type') == 'truefalse'])

        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 550)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=30, pady=25)

        types_title = tk.Label(setup_content,
                               text="Question Types",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        types_title.pack(anchor='w', pady=(0, 12))

        self.include_mc.set(True)
        self.include_input.set(True)
        self.include_tf.set(True)

        if mc_count > 0:
            self.create_sleek_checkbox(setup_content, f"Multiple Choice ({mc_count})", self.include_mc)
        if input_count > 0:
            self.create_sleek_checkbox(setup_content, f"Calculations ({input_count})", self.include_input)
        if tf_count > 0:
            self.create_sleek_checkbox(setup_content, f"True/False ({tf_count})", self.include_tf)

        note_label = tk.Label(setup_content,
                              text="ℹ️ Questions will be randomized automatically",
                              font=('SF Pro Display', 10, 'italic'),
                              bg=self.colors['card_bg'],
                              fg=self.colors['text_secondary'])
        note_label.pack(pady=(15, 0))

        button_container = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_container.pack(fill='x', pady=(20, 0))

        button_row = tk.Frame(button_container, bg=self.colors['card_bg'])
        button_row.pack()

        self.create_sleek_button(button_row, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=5)
        self.create_sleek_button(button_row, "Start Everything Test",
                                 lambda: self.start_configured_everything_test(all_questions), "special").pack(
            side='left', padx=5)

    def start_configured_everything_test(self, source_questions):
        filtered_questions = []

        for question in source_questions:
            question_type = question.get('type', 'multiple_choice')

            if question_type == 'multiple_choice' and self.include_mc.get():
                filtered_questions.append(question)
            elif question_type == 'input' and self.include_input.get():
                filtered_questions.append(question)
            elif question_type == 'truefalse' and self.include_tf.get():
                filtered_questions.append(question)

        if not filtered_questions:
            self.show_error_popup("Please select at least one question type!")
            return

        self.current_quiz_randomized = True
        random.shuffle(filtered_questions)

        self.current_questions = filtered_questions

        config_parts = []
        if not self.include_mc.get():
            config_parts.append("No MC")
        if not self.include_input.get():
            config_parts.append("No Calc")
        if not self.include_tf.get():
            config_parts.append("No T/F")

        config_parts.append("Random")

        if config_parts:
            self.quiz_name = f"Everything Test ({', '.join(config_parts)})"
        else:
            self.quiz_name = "Everything Test (Random)"

        self.reset_quiz_state()
        self.start_timer()
        self.show_question()

    def add_copy_menu(self, widget, text):
        def copy_text():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            original_bg = widget.cget('bg')
            widget.config(bg=self.colors['success'])
            self.root.after(200, lambda: widget.config(bg=original_bg))

        def copy_full_question():
            try:
                question_data = self.current_questions[self.current_question_index]
                question_type = question_data.get('type', 'multiple_choice')

                full_text = f"Question: {question_data['question']}\n\n"

                if question_type == 'multiple_choice':
                    if hasattr(self, 'current_shuffled_options') and self.current_shuffled_options:
                        for letter in ['a', 'b', 'c', 'd']:
                            if letter in self.current_shuffled_options:
                                option_text = self.current_shuffled_options[letter]
                                is_correct = (letter == self.current_correct_letter)
                                checkmark = " ✓" if is_correct else ""
                                full_text += f"{letter.upper()}. {option_text}{checkmark}\n"
                    else:
                        original_options = question_data['options']
                        correct_letter = question_data['correct']
                        for letter in ['a', 'b', 'c', 'd']:
                            if letter in original_options:
                                option_text = original_options[letter]
                                is_correct = (letter == correct_letter)
                                checkmark = " ✓" if is_correct else ""
                                full_text += f"{letter.upper()}. {option_text}{checkmark}\n"

                    if hasattr(self, 'current_correct_letter') and self.current_correct_letter:
                        full_text += f"\nCorrect Answer: {self.current_correct_letter.upper()}"
                    else:
                        full_text += f"\nCorrect Answer: {question_data['correct'].upper()}"

                elif question_type == 'truefalse':
                    correct_answer = question_data['correct']
                    true_mark = " ✓" if correct_answer == "TRUE" else ""
                    false_mark = " ✓" if correct_answer == "FALSE" else ""

                    full_text += f"TRUE{true_mark}\n"
                    full_text += f"FALSE{false_mark}\n"
                    full_text += f"\nCorrect Answer: {correct_answer}"

                elif question_type == 'input':
                    full_text += f"Type your answer:\n\n"
                    full_text += f"Correct Answer: {question_data['correct']}"

                self.root.clipboard_clear()
                self.root.clipboard_append(full_text)

                original_bg = widget.cget('bg')
                widget.config(bg=self.colors['primary'])
                self.root.after(300, lambda: widget.config(bg=original_bg))

            except Exception as e:
                copy_text()

        def copy_selected_text():
            try:
                selected = self.root.selection_get()
                if selected:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(selected)
                    return True
            except:
                pass
            return False

        def show_context_menu(event):
            context_menu = tk.Menu(self.root, tearoff=0,
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text'],
                                   activebackground=self.colors['primary'],
                                   activeforeground='white',
                                   bd=0)

            try:
                selected = self.root.selection_get()
                if selected and len(selected.strip()) > 0:
                    context_menu.add_command(label="📋 Copy Selected", command=copy_selected_text)
                    context_menu.add_separator()
            except:
                pass

            context_menu.add_command(label="📋 Copy This Text", command=copy_text)

            if hasattr(self, 'current_questions') and hasattr(self, 'current_question_index'):
                if self.current_question_index < len(self.current_questions):
                    context_menu.add_separator()
                    context_menu.add_command(label="📄 Copy Full Question", command=copy_full_question)

            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        widget.bind("<Button-3>", show_context_menu)

    def show_quiz_setup(self, quiz_name):
        self.clear_content()

        self.create_sleek_title(self.scrollable_frame, f"Configure {quiz_name}", 22)
        self.create_sleek_subtitle(self.scrollable_frame, "Choose question types and options", 11)

        quiz_questions = self.questions[quiz_name]

        mc_count = len([q for q in quiz_questions if q.get('type', 'multiple_choice') == 'multiple_choice'])
        input_count = len([q for q in quiz_questions if q.get('type') == 'input'])
        tf_count = len([q for q in quiz_questions if q.get('type') == 'truefalse'])

        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 550)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=30, pady=25)

        types_title = tk.Label(setup_content,
                               text="Question Types",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        types_title.pack(anchor='w', pady=(0, 12))

        self.include_mc.set(True)
        self.include_input.set(True)
        self.include_tf.set(True)
        self.randomize_questions.set(False)

        if mc_count > 0:
            self.create_sleek_checkbox(setup_content, f"Multiple Choice ({mc_count})", self.include_mc)
        if input_count > 0:
            self.create_sleek_checkbox(setup_content, f"Calculations ({input_count})", self.include_input)
        if tf_count > 0:
            self.create_sleek_checkbox(setup_content, f"True/False ({tf_count})", self.include_tf)

        separator = tk.Frame(setup_content, bg=self.colors['border'], height=1)
        separator.pack(fill='x', pady=15)

        options_title = tk.Label(setup_content,
                                 text="Options",
                                 font=('SF Pro Display', 12, 'normal'),
                                 bg=self.colors['card_bg'],
                                 fg=self.colors['text'])
        options_title.pack(anchor='w', pady=(0, 12))

        self.create_sleek_checkbox(setup_content, "Randomize order", self.randomize_questions)

        button_container = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_container.pack(fill='x', pady=(20, 0))

        button_row = tk.Frame(button_container, bg=self.colors['card_bg'])
        button_row.pack()

        self.create_sleek_button(button_row, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=5)
        self.create_sleek_button(button_row, "Start Quiz",
                                 lambda: self.start_configured_quiz(quiz_name, quiz_questions), "primary").pack(
            side='left', padx=5)

    def start_configured_quiz(self, quiz_name, source_questions):
        filtered_questions = []

        for question in source_questions:
            question_type = question.get('type', 'multiple_choice')

            if question_type == 'multiple_choice' and self.include_mc.get():
                filtered_questions.append(question)
            elif question_type == 'input' and self.include_input.get():
                filtered_questions.append(question)
            elif question_type == 'truefalse' and self.include_tf.get():
                filtered_questions.append(question)

        if not filtered_questions:
            self.show_error_popup("Please select at least one question type!")
            return

        self.current_quiz_randomized = self.randomize_questions.get()
        if self.current_quiz_randomized:
            random.shuffle(filtered_questions)

        self.current_questions = filtered_questions
        self.quiz_name = quiz_name

        config_parts = []
        if not self.include_mc.get():
            config_parts.append("No MC")
        if not self.include_input.get():
            config_parts.append("No Calc")
        if not self.include_tf.get():
            config_parts.append("No T/F")
        if self.randomize_questions.get():
            config_parts.append("Random")

        if config_parts:
            self.quiz_name += f" ({', '.join(config_parts)})"

        self.reset_quiz_state()
        self.show_question()

    def show_error_popup(self, message):
        overlay = tk.Toplevel(self.root)
        overlay.title("")
        overlay.geometry("350x150")
        overlay.configure(bg=self.colors['card_bg'])
        overlay.resizable(False, False)
        overlay.transient(self.root)
        overlay.grab_set()

        overlay.geometry("+%d+%d" % (self.root.winfo_rootx() + 300, self.root.winfo_rooty() + 200))

        error_label = tk.Label(overlay,
                               text=message,
                               font=('SF Pro Display', 11, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['danger'],
                               wraplength=300)
        error_label.pack(expand=True, pady=20)

        ok_btn = self.create_sleek_button(overlay, "OK", overlay.destroy, "primary")
        ok_btn.pack(pady=(0, 20))

    def show_question(self):
        if self.current_question_index >= len(self.current_questions):
            self.show_results()
            return

        self.clear_content()
        question_data = self.current_questions[self.current_question_index]

        progress_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        progress_container.pack(fill='x', pady=(15, 25))

        self.circular_progress = CircularProgress(progress_container, size=90)

        answered_questions = len(self.user_answers)
        correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
        wrong_count = answered_questions - correct_count

        self.circular_progress.update_progress(len(self.current_questions), correct_count, wrong_count)

        progress_text = f"Question {self.current_question_index + 1} of {len(self.current_questions)}"
        progress_label = tk.Label(progress_container,
                                  text=progress_text,
                                  font=('Segoe UI', 10, 'normal'),
                                  bg=self.colors['bg'],
                                  fg=self.colors['text_muted'])
        progress_label.pack(pady=(10, 0))

        if "Everything Test" in self.quiz_name and self.timer_running:
            self.timer_label = tk.Label(progress_container,
                                        text="⏱️ 00:00",
                                        font=('Segoe UI', 12, 'bold'),
                                        bg=self.colors['bg'],
                                        fg=self.colors['primary'])
            self.timer_label.pack(pady=(5, 0))

        question_card = self.create_ultra_modern_card(self.scrollable_frame, 750)
        question_content = tk.Frame(question_card, bg=self.colors['card_bg'])
        question_content.pack(fill='x', padx=35, pady=25)

        question_label = tk.Label(question_content,
                                  text=question_data['question'],
                                  font=('SF Pro Display', 13, 'normal'),
                                  bg=self.colors['card_bg'],
                                  fg=self.colors['text'],
                                  wraplength=680,
                                  justify='left')
        question_label.pack(pady=(0, 20))

        self.add_copy_menu(question_label, question_data['question'])

        self.setup_answer_input(question_content, question_data)

        self.feedback_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        self.feedback_frame.pack(fill='x', pady=12)

        nav_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        nav_container.pack(fill='x', pady=18)

        nav_frame = tk.Frame(nav_container, bg=self.colors['bg'])
        nav_frame.pack()

        left_nav = tk.Frame(nav_frame, bg=self.colors['bg'])
        left_nav.pack(side='left')

        self.create_sleek_button(left_nav, "Menu", self.show_main_menu, "ghost").pack(side='left', padx=2)
        self.create_sleek_button(left_nav, "🔄 Restart", self.restart_quiz, "ghost").pack(side='left', padx=2)

        self.submit_button = self.create_sleek_button(nav_frame, "Submit", self.submit_answer)
        self.submit_button.pack(side='right', padx=3)

    def setup_answer_input(self, parent, question_data):
        question_type = question_data.get('type', 'multiple_choice')

        if question_type == 'input':
            input_container = tk.Frame(parent, bg=self.colors['card_bg'])
            input_container.pack(pady=15)

            input_label = tk.Label(input_container,
                                   text="Your answer:",
                                   font=('SF Pro Display', 10, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text_secondary'])
            input_label.pack(pady=(0, 8))

            self.add_copy_menu(input_label, "Your answer:")

            self.text_input = tk.Entry(input_container,
                                       font=('SF Pro Display', 12, 'normal'),
                                       width=45,
                                       bg=self.colors['surface'],
                                       fg=self.colors['text'],
                                       insertbackground=self.colors['primary'],
                                       relief='flat',
                                       bd=0,
                                       highlightthickness=2,
                                       highlightcolor=self.colors['primary'],
                                       highlightbackground=self.colors['border'])
            self.text_input.pack(pady=6, ipady=10)
            self.text_input.focus()
            self.text_input.bind('<Return>', lambda e: self.submit_answer())

            self.add_copy_menu(input_label, question_data['question'])

            def copy_input_text():
                if hasattr(self, 'text_input') and self.text_input.get():
                    self.root.clipboard_clear()
                    self.root.clipboard_append(self.text_input.get())

            def show_input_context_menu(event):
                context_menu = tk.Menu(self.root, tearoff=0,
                                       bg=self.colors['card_bg'],
                                       fg=self.colors['text'],
                                       activebackground=self.colors['primary'],
                                       activeforeground='white',
                                       bd=0)
                context_menu.add_command(label="Copy Input", command=copy_input_text)
                try:
                    context_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    context_menu.grab_release()

            self.text_input.bind("<Button-3>", show_input_context_menu)

        elif question_type == 'truefalse':
            self.selected_answer.set("")

            tf_container = tk.Frame(parent, bg=self.colors['card_bg'])
            tf_container.pack(pady=18)

            tf_frame = tk.Frame(tf_container, bg=self.colors['card_bg'])
            tf_frame.pack()

            self.create_sleek_radiobutton(tf_frame, "TRUE", self.selected_answer, "TRUE")
            self.create_sleek_radiobutton(tf_frame, "FALSE", self.selected_answer, "FALSE")

        else:
            self.selected_answer.set("")

            original_options = question_data['options']
            option_letters = ['a', 'b', 'c', 'd']
            option_texts = [original_options[letter] for letter in option_letters]

            shuffled_texts = option_texts.copy()
            random.shuffle(shuffled_texts)

            self.current_shuffled_options = {}
            original_correct_text = original_options[question_data['correct']]

            for i, letter in enumerate(option_letters):
                self.current_shuffled_options[letter] = shuffled_texts[i]
                if shuffled_texts[i] == original_correct_text:
                    self.current_correct_letter = letter

            options_container = tk.Frame(parent, bg=self.colors['card_bg'])
            options_container.pack(pady=15)

            for letter in option_letters:
                text = self.current_shuffled_options[letter]
                option_text = f"{letter.upper()}. {text}"
                self.create_sleek_radiobutton(options_container, option_text, self.selected_answer, letter)

    def submit_answer(self):
        question_data = self.current_questions[self.current_question_index]
        question_type = question_data.get('type', 'multiple_choice')

        if question_type == 'input':
            if not hasattr(self, 'text_input') or not self.text_input.get().strip():
                return
            user_answer = self.text_input.get().strip()
        else:
            user_answer = self.selected_answer.get()
            if not user_answer:
                return

        if self.feedback_visible:
            self.next_question()
            return

        if question_type == 'multiple_choice':
            correct_answer = self.current_correct_letter
            is_correct = user_answer == correct_answer
            display_options = self.current_shuffled_options
        else:
            correct_answer = question_data['correct']
            if question_type == 'input':
                user_normalized = user_answer.lower().strip()
                correct_normalized = correct_answer.lower().strip()
                is_correct = user_normalized == correct_normalized
            else:
                is_correct = user_answer == correct_answer
            display_options = None

        self.user_answers.append({
            'question': question_data['question'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'options': display_options,
            'is_correct': is_correct,
            'type': question_type
        })

        if is_correct:
            self.score += 1

        self.show_feedback(is_correct, correct_answer, display_options, question_type)

    def show_feedback(self, is_correct, correct_answer, options, question_type):
        self.feedback_visible = True

        self.submit_button.config(text="Continue")

        answered_questions = len(self.user_answers)
        correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
        wrong_count = answered_questions - correct_count
        self.circular_progress.update_progress(len(self.current_questions), correct_count, wrong_count)

        for widget in self.feedback_frame.winfo_children():
            widget.destroy()

        feedback_card = self.create_ultra_modern_card(self.feedback_frame, 650, minimal=True)
        feedback_content = tk.Frame(feedback_card, bg=self.colors['surface'])
        feedback_content.pack(fill='x', padx=20, pady=12)

        if is_correct:
            icon = "✓"
            message = "Correct"
            color = self.colors['success']
        else:
            icon = "×"
            message = "Incorrect"
            color = self.colors['danger']

        feedback_header = tk.Label(feedback_content,
                                   text=f"{icon} {message}",
                                   font=('SF Pro Display', 12, 'normal'),
                                   bg=self.colors['surface'],
                                   fg=color)
        feedback_header.pack()

        self.add_copy_menu(feedback_header, f"{icon} {message}")

        if not is_correct:
            if question_type == 'multiple_choice':
                correct_text = f"{correct_answer.upper()}. {options[correct_answer]}"
            else:
                correct_text = f"{correct_answer}"

            correct_label = tk.Label(feedback_content,
                                     text=correct_text,
                                     font=('SF Pro Display', 9, 'normal'),
                                     bg=self.colors['surface'],
                                     fg=self.colors['text_secondary'],
                                     wraplength=600)
            correct_label.pack(pady=(4, 0))

            self.add_copy_menu(correct_label, correct_text)

        self.submit_button.config(text="Continue")

    def next_question(self):
        self.feedback_visible = False
        self.current_question_index += 1
        self.show_question()

    def show_results(self):
        self.clear_content()

        final_time = self.stop_timer()

        self.canvas.yview_moveto(0)
        self.root.update_idletasks()

        self.create_sleek_title(self.scrollable_frame, "Complete", 26)
        self.create_sleek_subtitle(self.scrollable_frame, f"{self.quiz_name}")

        score_card = self.create_ultra_modern_card(self.scrollable_frame, 500, minimal=True)
        score_content = tk.Frame(score_card, bg=self.colors['surface'])
        score_content.pack(fill='x', padx=30, pady=25)

        percentage = (self.score / len(self.current_questions)) * 100 if len(self.current_questions) > 0 else 0

        score_text = tk.Label(score_content,
                              text=f"{self.score}/{len(self.current_questions)}",
                              font=('Segoe UI', 28, 'bold'),
                              bg=self.colors['surface'],
                              fg=self.colors['primary'])
        score_text.pack()

        percentage_text = tk.Label(score_content,
                                   text=f"{percentage:.0f}%",
                                   font=('Segoe UI', 16, 'normal'),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text'])
        percentage_text.pack(pady=(5, 10))

        if "Everything Test" in self.quiz_name and final_time > 0:
            time_text = tk.Label(score_content,
                                 text=f"⏱️ Time: {self.format_time(final_time)}",
                                 font=('Segoe UI', 14, 'normal'),
                                 bg=self.colors['surface'],
                                 fg=self.colors['warning'])
            time_text.pack(pady=(0, 15))

        if percentage >= 90:
            message = "Excellent"
            color = self.colors['success']
        elif percentage >= 80:
            message = "Great"
            color = self.colors['success']
        elif percentage >= 70:
            message = "Good"
            color = self.colors['warning']
        elif percentage >= 60:
            message = "Fair"
            color = self.colors['warning']
        else:
            message = "Review"
            color = self.colors['danger']

        message_label = tk.Label(score_content,
                                 text=message,
                                 font=('Segoe UI', 11, 'normal'),
                                 bg=self.colors['surface'],
                                 fg=color)
        message_label.pack()

        button_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        button_container.pack(fill='x', pady=(30, 20))

        button_frame = tk.Frame(button_container, bg=self.colors['bg'])
        button_frame.pack(anchor='center')

        restart_btn = self.create_sleek_button(button_frame, "🔄 Restart", self.restart_quiz, "secondary")
        restart_btn.pack(side='left', padx=8)

        menu_btn = self.create_sleek_button(button_frame, "📋 Menu", self.show_main_menu, "primary")
        menu_btn.pack(side='left', padx=8)

        self.root.update_idletasks()

    def show_random_setup(self):
        self.clear_content()

        self.create_sleek_title(self.scrollable_frame, "Random Quiz", 22)

        all_questions = []
        for questions in self.questions.values():
            all_questions.extend(questions)

        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 400)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=25, pady=20)

        instruction = tk.Label(setup_content,
                               text=f"Questions (max {len(all_questions)})",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        instruction.pack(pady=(0, 12))

        input_frame = tk.Frame(setup_content, bg=self.colors['card_bg'])
        input_frame.pack(pady=8)

        self.random_count_var = tk.StringVar(value=str(min(20, len(all_questions))))
        count_entry = tk.Entry(input_frame,
                               textvariable=self.random_count_var,
                               font=('SF Pro Display', 11, 'normal'),
                               width=8,
                               justify='center',
                               bg=self.colors['surface'],
                               fg=self.colors['text'],
                               relief='flat',
                               bd=0,
                               highlightthickness=1,
                               highlightcolor=self.colors['primary'],
                               highlightbackground=self.colors['border'])
        count_entry.pack(pady=6, ipady=6)

        button_frame = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_frame.pack(pady=(15, 0))

        self.create_sleek_button(button_frame, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=3)
        self.create_sleek_button(button_frame, "Start", lambda: self.start_random_quiz(all_questions),
                                 "primary").pack(side='left', padx=3)

    def start_random_quiz(self, all_questions):
        try:
            count = int(self.random_count_var.get())
            if 1 <= count <= len(all_questions):
                selected_questions = random.sample(all_questions, count)
                self.current_questions = selected_questions
                self.quiz_name = f"Random ({count} questions)"
                self.reset_quiz_state()
                self.show_question()
            else:
                self.show_error_popup(f"Enter 1-{len(all_questions)}")
        except ValueError:
            self.show_error_popup("Enter a valid number")

    def restart_quiz(self):
        original_questions = self.current_questions.copy()

        if self.current_quiz_randomized:
            random.shuffle(original_questions)
            self.current_questions = original_questions

        self.current_question_index = 0
        self.score = 0
        self.user_answers = []
        self.selected_answer.set("")
        self.feedback_visible = False
        self.current_shuffled_options = {}
        self.current_correct_letter = None

        if hasattr(self, 'text_input'):
            try:
                self.text_input.delete(0, tk.END)
            except:
                pass

        if "Everything Test" in self.quiz_name:
            self.start_timer()

        self.show_question()

    def reset_quiz_state(self):
        self.current_question_index = 0
        self.score = 0
        self.user_answers = []
        self.selected_answer.set("")
        self.feedback_visible = False
        self.current_shuffled_options = {}
        self.current_correct_letter = None

    def run(self):
        self.root.mainloop()


def main():
    try:
        app = UltraModernQuizApp()
        app.run()
    except ImportError:
        print("Error: Could not import questions. Please ensure questions.py is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()