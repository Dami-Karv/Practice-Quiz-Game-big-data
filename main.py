# main.py
# Ultra-modern, minimalist tkinter quiz with question filtering and sleek design

import tkinter as tk
from tkinter import ttk
import random
import math
from questions import QUIZ_QUESTIONS


class CircularProgress:
    """Modern circular progress indicator showing correct/wrong answers"""

    def __init__(self, parent, size=80):
        self.size = size
        self.canvas = tk.Canvas(parent, width=size, height=size, bg='#0a0a0f', highlightthickness=0)
        self.canvas.pack()

        # Colors
        self.bg_color = '#1e293b'
        self.correct_color = '#10b981'
        self.wrong_color = '#ef4444'

        self.total_questions = 0
        self.correct_answers = 0
        self.wrong_answers = 0

    def update_progress(self, total, correct, wrong):
        """Update the circular progress"""
        self.total_questions = total
        self.correct_answers = correct
        self.wrong_answers = wrong

        # Clear canvas
        self.canvas.delete("all")

        # Calculate angles
        if total > 0:
            correct_angle = (correct / total) * 360
            wrong_angle = (wrong / total) * 360
        else:
            correct_angle = wrong_angle = 0

        center = self.size // 2
        radius = (self.size // 2) - 10

        # Background circle
        self.canvas.create_oval(center - radius, center - radius,
                                center + radius, center + radius,
                                outline=self.bg_color, width=6, fill="")

        # Correct answers arc (green)
        if correct_angle > 0:
            self.canvas.create_arc(center - radius, center - radius,
                                   center + radius, center + radius,
                                   start=90, extent=-correct_angle,
                                   outline=self.correct_color, width=6,
                                   style="arc")

        # Wrong answers arc (red)
        if wrong_angle > 0:
            self.canvas.create_arc(center - radius, center - radius,
                                   center + radius, center + radius,
                                   start=90 - correct_angle, extent=-wrong_angle,
                                   outline=self.wrong_color, width=6,
                                   style="arc")

        # Percentage text in center
        if total > 0:
            percentage = round((correct / total) * 100)
            self.canvas.create_text(center, center - 5,
                                    text=f"{percentage}%",
                                    font=('Segoe UI', 12, 'bold'),
                                    fill='#f8fafc')

            # Score text below percentage
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

        # Ultra-modern minimalist colors
        self.colors = {
            'bg': '#0a0a0f',  # Deep space black
            'card_bg': '#1a1a2e',  # Dark navy cards
            'accent': '#16213e',  # Subtle accent
            'primary': '#6366f1',  # Modern indigo
            'secondary': '#8b5cf6',  # Purple accent
            'success': '#10b981',  # Clean emerald
            'danger': '#ef4444',  # Clean red
            'warning': '#f59e0b',  # Clean amber
            'text': '#f8fafc',  # Almost white
            'text_secondary': '#94a3b8',  # Cool gray
            'text_muted': '#64748b',  # Muted gray
            'surface': '#0f172a',  # Dark surface
            'border': '#1e293b'  # Subtle border
        }

        # Quiz data
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
        self.used_questions = []

        # Question filtering options
        self.include_mc = tk.BooleanVar(value=True)
        self.include_input = tk.BooleanVar(value=True)
        self.include_tf = tk.BooleanVar(value=True)
        self.randomize_questions = tk.BooleanVar(value=False)

        # UI components
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

        # Bind window resize
        self.root.bind('<Configure>', self.on_window_resize)
        self.root.bind('<KeyPress>', self.handle_keypress)

    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        """Setup ultra-modern responsive UI structure"""
        # Main container with ultra-minimal padding
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill='both', expand=True, padx=40, pady=20)

        # Scrollable canvas with minimal styling
        self.canvas = tk.Canvas(self.main_container, bg=self.colors['bg'], highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack with minimal visual elements
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Configure canvas for perfect centering
        self.canvas.bind('<Configure>', self.configure_canvas_width)

        # Smooth mouse wheel scrolling
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", on_mousewheel)

    def configure_canvas_width(self, event=None):
        """Configure canvas to center content perfectly"""
        if event:
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas.find_all()[0], width=canvas_width)

    def handle_keypress(self, event):
        """Handle key presses, specifically space bar for submit"""
        if event.keysym == 'space':
            # Only submit if we're not typing in a text input field
            focused_widget = self.root.focus_get()
            if not isinstance(focused_widget, tk.Entry):
                self.submit_answer()
                return 'break'  # Prevent the space from being processed further

    def on_window_resize(self, event=None):
        """Handle window resize for ultra-responsive design"""
        if event and event.widget == self.root:
            width = self.root.winfo_width()
            padding = max(20, min(80, (width - 900) // 4))
            self.main_container.configure(padx=padding)

    def clear_content(self):
        """Clear all content"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def create_ultra_modern_card(self, parent, width=700, minimal=False):
        """Create ultra-modern minimalist card"""
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
        """Create ultra-sleek title"""
        container = tk.Frame(parent, bg=self.colors['bg'])
        container.pack(fill='x', pady=(0, 8))

        title = tk.Label(container,
                         text=text,
                         font=('SF Pro Display', size, 'normal'),  # Ultra-light weight
                         bg=self.colors['bg'],
                         fg=color or self.colors['text'])
        title.pack()
        return title

    def create_sleek_subtitle(self, parent, text, size=11, color=None):
        """Create minimalist subtitle"""
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
        """Start the quiz timer"""
        import time
        self.timer_start = time.time()
        self.timer_running = True
        self.elapsed_time = 0
        self.update_timer_display()

    def stop_timer(self):
        """Stop the quiz timer"""
        if self.timer_running:
            import time
            self.elapsed_time = time.time() - self.timer_start
            self.timer_running = False
            if self.timer_after_id:
                self.root.after_cancel(self.timer_after_id)
        return self.elapsed_time

    def update_timer_display(self):
        """Update timer display every second"""
        if self.timer_running and self.timer_start:
            import time
            current_time = time.time() - self.timer_start
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)

            if self.timer_label:
                self.timer_label.config(text=f"⏱️ {minutes:02d}:{seconds:02d}")

            # Schedule next update
            self.timer_after_id = self.root.after(1000, self.update_timer_display)

    def format_time(self, seconds):
        """Format seconds into MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def save_to_leaderboard(self, name, percentage, time_taken):
        """Save score to online leaderboard using JSONBin.io"""
        import requests
        import json
        from datetime import datetime

        try:
            # First, get existing leaderboard
            scores = self.load_leaderboard()

            # Add new score
            new_score = {
                "name": name,
                "score": percentage,
                "time": time_taken,
                "questions": len(self.current_questions),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "timestamp": datetime.now().timestamp()
            }

            scores.append(new_score)

            # Sort by score (descending), then by time (ascending)
            scores.sort(key=lambda x: (-x["score"], x["time"]))

            # Keep top 100 scores
            scores = scores[:100]

            # Save to JSONBin.io
            bin_id = "67472e1ead19ca34f8c8f8e2"  # Shared global leaderboard
            url = f"https://api.jsonbin.io/v3/b/{bin_id}"

            headers = {
                'Content-Type': 'application/json',
            }

            response = requests.put(url, json=scores, headers=headers, timeout=10)

            if response.status_code == 200:
                return True
            else:
                # Fallback to local storage if online fails
                return self.save_to_local_leaderboard(scores)

        except Exception as e:
            print(f"Error saving to online leaderboard: {e}")
            # Fallback to local storage
            return self.save_to_local_leaderboard(scores if 'scores' in locals() else [])

    def load_leaderboard(self):
        """Load leaderboard from online source (JSONBin.io) with local fallback"""
        import requests
        import json
        import os

        try:
            # Try to load from online first
            bin_id = "67472e1ead19ca34f8c8f8e2"  # Same bin ID
            url = f"https://api.jsonbin.io/v3/b/{bin_id}/latest"

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'record' in data:
                    return data['record']
                return []
            else:
                # Fallback to local
                return self.load_local_leaderboard()

        except Exception as e:
            print(f"Error loading online leaderboard: {e}")
            # Fallback to local
            return self.load_local_leaderboard()

    def save_to_local_leaderboard(self, scores):
        """Backup method - save to local file"""
        import json
        import os

        try:
            leaderboard_file = "leaderboard_backup.json"
            with open(leaderboard_file, 'w') as f:
                json.dump(scores, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving to local leaderboard: {e}")
            return False

    def load_local_leaderboard(self):
        """Backup method - load from local file"""
        import json
        import os

        try:
            leaderboard_file = "leaderboard_backup.json"
            if os.path.exists(leaderboard_file):
                with open(leaderboard_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading local leaderboard: {e}")
            return []

    def submit_to_leaderboard(self, percentage, time_taken):
        """Submit score to leaderboard"""
        # Simple popup to get player name
        name_window = tk.Toplevel(self.root)
        name_window.title("Submit Score")
        name_window.geometry("400x200")
        name_window.configure(bg=self.colors['card_bg'])
        name_window.transient(self.root)
        name_window.grab_set()

        # Center the popup
        name_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 300, self.root.winfo_rooty() + 200))

        tk.Label(name_window, text="Enter your name:",
                 font=('Segoe UI', 12, 'normal'),
                 bg=self.colors['card_bg'], fg=self.colors['text']).pack(pady=20)

        name_entry = tk.Entry(name_window, font=('Segoe UI', 12, 'normal'), width=20)
        name_entry.pack(pady=10)
        name_entry.focus()

        def submit_score():
            name = name_entry.get().strip()
            if name:
                success = self.save_to_leaderboard(name, percentage, time_taken)
                name_window.destroy()
                if success:
                    self.show_success_popup("Score submitted successfully!")
                else:
                    self.show_error_popup("Failed to submit score. Try again later.")

        tk.Button(name_window, text="Submit", command=submit_score,
                  bg=self.colors['primary'], fg='white',
                  font=('Segoe UI', 10, 'normal')).pack(pady=10)

        name_entry.bind('<Return>', lambda e: submit_score())

    def show_leaderboard(self):
        """Display leaderboard in a new window"""
        leaderboard_window = tk.Toplevel(self.root)
        leaderboard_window.title("🏆 Leaderboard")
        leaderboard_window.geometry("650x550")
        leaderboard_window.configure(bg=self.colors['bg'])

        # Title with online status
        title = tk.Label(leaderboard_window, text="🏆 Everything Test Leaderboard",
                         font=('Segoe UI', 16, 'bold'),
                         bg=self.colors['bg'], fg=self.colors['text'])
        title.pack(pady=10)

        # Online status
        status_label = tk.Label(leaderboard_window, text="🌐 Global Online Leaderboard",
                                font=('Segoe UI', 10, 'normal'),
                                bg=self.colors['bg'], fg=self.colors['success'])
        status_label.pack(pady=(0, 10))

        # Refresh button
        refresh_btn = tk.Button(leaderboard_window, text="🔄 Refresh",
                                command=lambda: self.refresh_leaderboard(scrollable_frame),
                                bg=self.colors['primary'], fg='white',
                                font=('Segoe UI', 9, 'normal'))
        refresh_btn.pack(pady=5)

        # Scrollable frame for scores
        canvas = tk.Canvas(leaderboard_window, bg=self.colors['bg'])
        scrollbar = tk.Scrollbar(leaderboard_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Load and display scores
        self.display_scores(scrollable_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Close button
        close_btn = tk.Button(leaderboard_window, text="Close",
                              command=leaderboard_window.destroy,
                              bg=self.colors['secondary'], fg='white')
        close_btn.pack(pady=20)

    def display_scores(self, parent):
        """Display scores in the leaderboard"""
        # Clear existing scores
        for widget in parent.winfo_children():
            widget.destroy()

        scores = self.load_leaderboard()

        if scores:
            # Add header
            header_frame = tk.Frame(parent, bg=self.colors['surface'], relief='raised', bd=2)
            header_frame.pack(fill='x', padx=20, pady=(10, 5))

            header_label = tk.Label(header_frame,
                                    text="🏆 Perfect Scores Only (Randomized Everything Test)",
                                    font=('Segoe UI', 10, 'bold'),
                                    bg=self.colors['surface'], fg=self.colors['primary'],
                                    anchor='center')
            header_label.pack(fill='x', padx=15, pady=8)

            for i, score in enumerate(scores[:25]):  # Top 25
                rank = i + 1
                if rank == 1:
                    emoji = "🥇"
                    color = self.colors['warning']
                elif rank == 2:
                    emoji = "🥈"
                    color = self.colors['text_secondary']
                elif rank == 3:
                    emoji = "🥉"
                    color = self.colors['warning']
                else:
                    emoji = f"{rank}."
                    color = self.colors['text']

                score_frame = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=1)
                score_frame.pack(fill='x', padx=20, pady=2)

                # Score text with better formatting
                score_text = f"{emoji} {score['name']}"
                time_text = f"⏱️ {self.format_time(score['time'])} • {score['questions']} questions"

                name_label = tk.Label(score_frame, text=score_text,
                                      font=('Segoe UI', 11, 'bold'),
                                      bg=self.colors['card_bg'], fg=color,
                                      anchor='w')
                name_label.pack(fill='x', padx=15, pady=(8, 2))

                details_label = tk.Label(score_frame, text=time_text,
                                         font=('Segoe UI', 9, 'normal'),
                                         bg=self.colors['card_bg'], fg=self.colors['text_secondary'],
                                         anchor='w')
                details_label.pack(fill='x', padx=15, pady=(0, 8))
        else:
            no_scores = tk.Label(parent, text="No perfect scores yet! 🎯\nGet 100% on randomized Everything Test!",
                                 font=('Segoe UI', 14, 'normal'),
                                 bg=self.colors['bg'], fg=self.colors['text_secondary'],
                                 justify='center')
            no_scores.pack(pady=50)


    def refresh_leaderboard(self, parent):
        """Refresh the leaderboard display"""
        self.display_scores(parent)

    def show_success_popup(self, message):
        """Show success message"""
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
        """Create ultra-modern sleek button"""
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
        else:  # secondary
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

        # Add hover effect
        def on_enter(e):
            button.config(bg=hover_bg)

        def on_leave(e):
            button.config(bg=bg_color)

        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)

        return button

    def create_sleek_checkbox(self, parent, text, variable):
        """Create modern checkbox with sleek styling"""
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

        # Hover effect
        def on_enter(e):
            checkbox.config(bg=self.colors['accent'])

        def on_leave(e):
            checkbox.config(bg=self.colors['card_bg'])

        checkbox.bind('<Enter>', on_enter)
        checkbox.bind('<Leave>', on_leave)

        return checkbox

    def create_sleek_radiobutton(self, parent, text, variable, value):
        """Create ultra-modern radio button with perfect visibility"""
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

        # Hover effect for better UX
        def on_enter(e):
            radio.config(bg=self.colors['accent'])

        def on_leave(e):
            radio.config(bg=self.colors['card_bg'])

        radio.bind('<Enter>', on_enter)
        radio.bind('<Leave>', on_leave)

        # Add copy functionality for answer options
        self.add_copy_menu(radio, text)

        return radio

    def show_main_menu(self):
        """Display ultra-modern main menu"""
        self.clear_content()

        # Ultra-sleek header with minimal styling
        self.create_sleek_title(self.scrollable_frame, "485 Quiz", 32)
        self.create_sleek_subtitle(self.scrollable_frame, "Machine Learning Assessment Platform", 12)

        # Minimal stats overview
        stats_card = self.create_ultra_modern_card(self.scrollable_frame, 550, minimal=True)
        stats_content = tk.Frame(stats_card, bg=self.colors['surface'])
        stats_content.pack(fill='x', padx=25, pady=16)

        total_questions = sum(len(questions) for questions in self.questions.values())
        available_questions = total_questions - len(self.used_questions)

        stats_text = f"{len(self.questions)} Quizzes • {available_questions}/{total_questions} Available"
        stats_label = tk.Label(stats_content,
                               text=stats_text,
                               font=('SF Pro Display', 11, 'normal'),
                               bg=self.colors['surface'],
                               fg=self.colors['primary'])
        stats_label.pack()

        # Sleek quiz cards with minimal design
        descriptions = {
            "Quiz #1": "NumPy • pandas • matplotlib • statistics",
            "Quiz #2": "PCA • neural networks • ML algorithms",
            "Quiz #3": "CNNs • RNNs • transformers • vision",
            "Quiz #4": "NLP • video analysis • reinforcement learning"
        }

        for quiz_name, questions in self.questions.items():
            available_in_quiz = len([q for q in questions if q not in self.used_questions])

            card = self.create_ultra_modern_card(self.scrollable_frame, 650)
            content = tk.Frame(card, bg=self.colors['card_bg'])
            content.pack(fill='x', padx=28, pady=18)

            # Minimal quiz title
            title_label = tk.Label(content,
                                   text=quiz_name,
                                   font=('SF Pro Display', 15, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text'],
                                   anchor='w')
            title_label.pack(fill='x', pady=(0, 4))

            # Minimal description
            if quiz_name in descriptions:
                desc_label = tk.Label(content,
                                      text=descriptions[quiz_name],
                                      font=('SF Pro Display', 9, 'normal'),
                                      bg=self.colors['card_bg'],
                                      fg=self.colors['text_secondary'],
                                      anchor='w')
                desc_label.pack(fill='x', pady=(0, 12))

            # Clean bottom row
            bottom_frame = tk.Frame(content, bg=self.colors['card_bg'])
            bottom_frame.pack(fill='x')

            count_label = tk.Label(bottom_frame,
                                   text=f"{available_in_quiz}/{len(questions)}",
                                   font=('SF Pro Display', 8, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text_muted'])
            count_label.pack(side='left')

            start_btn = self.create_sleek_button(bottom_frame,
                                                 "Configure →" if available_in_quiz > 0 else "Complete",
                                                 lambda name=quiz_name: self.show_quiz_setup(name),
                                                 "primary" if available_in_quiz > 0 else "ghost")
            if available_in_quiz == 0:
                start_btn.config(state='disabled')
            start_btn.pack(side='right')

        # Special Everything Test card
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
                                    text=f"{available_questions} remaining",
                                    font=('SF Pro Display', 8, 'normal'),
                                    bg=self.colors['card_bg'],
                                    fg=self.colors['text_muted'])
        everything_count.pack(side='left')

        everything_btn = self.create_sleek_button(everything_bottom,
                                                  "Configure →" if available_questions > 0 else "Reset",
                                                  self.show_everything_setup if available_questions > 0 else self.reset_used_questions,
                                                  "special" if available_questions > 0 else "secondary")
        everything_btn.pack(side='right')

        # Minimal footer actions
        footer_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        footer_container.pack(fill='x', pady=25)

        footer_frame = tk.Frame(footer_container, bg=self.colors['bg'])
        footer_frame.pack()

        self.create_sleek_button(footer_frame, "Random", self.show_random_setup, "ghost").pack(side='left', padx=3)
        self.create_sleek_button(footer_frame, "Reset", self.reset_used_questions, "ghost").pack(side='left', padx=3)
        self.create_sleek_button(footer_frame, "Exit", self.root.quit, "ghost").pack(side='left', padx=3)

    def show_everything_setup(self):
        """Show Everything Test setup with question type filtering"""
        self.clear_content()

        # Clean header
        self.create_sleek_title(self.scrollable_frame, "Configure Everything Test", 22)
        self.create_sleek_subtitle(self.scrollable_frame, "Choose question types from all quizzes", 11)

        # Get all available questions
        all_questions = []
        for questions in self.questions.values():
            all_questions.extend(questions)
        available_questions = [q for q in all_questions if q not in self.used_questions]

        # Count question types
        mc_count = len([q for q in available_questions if q.get('type', 'multiple_choice') == 'multiple_choice'])
        input_count = len([q for q in available_questions if q.get('type') == 'input'])
        tf_count = len([q for q in available_questions if q.get('type') == 'truefalse'])

        if not available_questions:
            # No questions available
            error_card = self.create_ultra_modern_card(self.scrollable_frame, 500)
            error_content = tk.Frame(error_card, bg=self.colors['card_bg'])
            error_content.pack(fill='x', padx=30, pady=25)

            error_label = tk.Label(error_content,
                                   text="All questions completed ✓",
                                   font=('SF Pro Display', 14, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['success'])
            error_label.pack(pady=15)

            back_btn = self.create_sleek_button(error_content, "← Back", self.show_main_menu, "secondary")
            back_btn.pack()
            return

        # Question type selection card
        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 550)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=30, pady=25)

        # Section title
        types_title = tk.Label(setup_content,
                               text="Question Types",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        types_title.pack(anchor='w', pady=(0, 12))

        # Reset filters
        # Reset filters
        self.include_mc.set(True)
        self.include_input.set(True)
        self.include_tf.set(True)

        # Question type checkboxes
        if mc_count > 0:
            self.create_sleek_checkbox(setup_content, f"Multiple Choice ({mc_count})", self.include_mc)
        if input_count > 0:
            self.create_sleek_checkbox(setup_content, f"Calculations ({input_count})", self.include_input)
        if tf_count > 0:
            self.create_sleek_checkbox(setup_content, f"True/False ({tf_count})", self.include_tf)

        # Add note about randomization
        note_label = tk.Label(setup_content,
                              text="ℹ️ Questions will be randomized automatically",
                              font=('SF Pro Display', 10, 'italic'),
                              bg=self.colors['card_bg'],
                              fg=self.colors['text_secondary'])
        note_label.pack(pady=(15, 0))

        # Action buttons
        button_container = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_container.pack(fill='x', pady=(20, 0))

        # Button row
        button_row = tk.Frame(button_container, bg=self.colors['card_bg'])
        button_row.pack()

        self.create_sleek_button(button_row, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=5)
        self.create_sleek_button(button_row, "Start Everything Test",
                                 lambda: self.start_configured_everything_test(available_questions), "special").pack(
            side='left', padx=5)

    def start_configured_everything_test(self, source_questions):
        """Start Everything Test with configured options"""
        # Filter questions based on selected types
        filtered_questions = []

        for question in source_questions:
            question_type = question.get('type', 'multiple_choice')

            if question_type == 'multiple_choice' and self.include_mc.get():
                filtered_questions.append(question)
            elif question_type == 'input' and self.include_input.get():
                filtered_questions.append(question)
            elif question_type == 'truefalse' and self.include_tf.get():
                filtered_questions.append(question)

        # Check if any questions are selected
        if not filtered_questions:
            self.show_error_popup("Please select at least one question type!")
            return

        # Always randomize Everything Test
        self.current_quiz_randomized = True
        random.shuffle(filtered_questions)

        # Set up the quiz
        self.current_questions = filtered_questions

        # Add configuration info to quiz name
        config_parts = []
        if not self.include_mc.get():
            config_parts.append("No MC")
        if not self.include_input.get():
            config_parts.append("No Calc")
        if not self.include_tf.get():
            config_parts.append("No T/F")

        config_parts.append("Random")  # Always add Random since it's always randomized

        if config_parts:
            self.quiz_name = f"Everything Test ({', '.join(config_parts)})"
        else:
            self.quiz_name = "Everything Test (Random)"

        self.reset_quiz_state()
        self.start_timer()
        self.show_question()

    def add_copy_menu(self, widget, text):
        """Add right-click context menu for copying text with full question option"""

        def copy_text():
            """Copy just this element's text"""
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            # Brief visual feedback
            original_bg = widget.cget('bg')
            widget.config(bg=self.colors['success'])
            self.root.after(200, lambda: widget.config(bg=original_bg))

        def copy_full_question():
            """Copy the entire question with all options and correct answer"""
            try:
                question_data = self.current_questions[self.current_question_index]
                question_type = question_data.get('type', 'multiple_choice')

                # Start with the question
                full_text = f"Question: {question_data['question']}\n\n"

                if question_type == 'multiple_choice':
                    # Add all options with correct answer marked
                    if hasattr(self, 'current_shuffled_options') and self.current_shuffled_options:
                        # Use the shuffled options as displayed
                        for letter in ['a', 'b', 'c', 'd']:
                            if letter in self.current_shuffled_options:
                                option_text = self.current_shuffled_options[letter]
                                is_correct = (letter == self.current_correct_letter)
                                checkmark = " ✓" if is_correct else ""
                                full_text += f"{letter.upper()}. {option_text}{checkmark}\n"
                    else:
                        # Use original options if shuffled not available
                        original_options = question_data['options']
                        correct_letter = question_data['correct']
                        for letter in ['a', 'b', 'c', 'd']:
                            if letter in original_options:
                                option_text = original_options[letter]
                                is_correct = (letter == correct_letter)
                                checkmark = " ✓" if is_correct else ""
                                full_text += f"{letter.upper()}. {option_text}{checkmark}\n"

                    # Add correct answer
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

                # Copy to clipboard
                self.root.clipboard_clear()
                self.root.clipboard_append(full_text)

                # Visual feedback
                original_bg = widget.cget('bg')
                widget.config(bg=self.colors['primary'])
                self.root.after(300, lambda: widget.config(bg=original_bg))

            except Exception as e:
                # Fallback to simple text copy
                copy_text()

        def copy_selected_text():
            """Copy currently selected text if any"""
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
            """Show enhanced context menu"""
            context_menu = tk.Menu(self.root, tearoff=0,
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text'],
                                   activebackground=self.colors['primary'],
                                   activeforeground='white',
                                   bd=0)

            # Check if there's selected text
            try:
                selected = self.root.selection_get()
                if selected and len(selected.strip()) > 0:
                    context_menu.add_command(label="📋 Copy Selected", command=copy_selected_text)
                    context_menu.add_separator()
            except:
                pass

            # Add copy options
            context_menu.add_command(label="📋 Copy This Text", command=copy_text)

            # Add full question copy if we're in a question
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
        """Show quiz setup with question type filtering"""
        self.clear_content()

        # Clean header
        self.create_sleek_title(self.scrollable_frame, f"Configure {quiz_name}", 22)
        self.create_sleek_subtitle(self.scrollable_frame, "Choose question types and options", 11)

        # Get question counts for this quiz
        quiz_questions = self.questions[quiz_name]
        available_questions = [q for q in quiz_questions if q not in self.used_questions]

        mc_count = len([q for q in available_questions if q.get('type', 'multiple_choice') == 'multiple_choice'])
        input_count = len([q for q in available_questions if q.get('type') == 'input'])
        tf_count = len([q for q in available_questions if q.get('type') == 'truefalse'])

        if not available_questions:
            # No questions available
            error_card = self.create_ultra_modern_card(self.scrollable_frame, 500)
            error_content = tk.Frame(error_card, bg=self.colors['card_bg'])
            error_content.pack(fill='x', padx=30, pady=25)

            error_label = tk.Label(error_content,
                                   text="All questions completed ✓",
                                   font=('SF Pro Display', 14, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['success'])
            error_label.pack(pady=15)

            back_btn = self.create_sleek_button(error_content, "← Back", self.show_main_menu, "secondary")
            back_btn.pack()
            return

        # Question type selection card
        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 550)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=30, pady=25)

        # Section title
        types_title = tk.Label(setup_content,
                               text="Question Types",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        types_title.pack(anchor='w', pady=(0, 12))

        # Reset filters
        self.include_mc.set(True)
        self.include_input.set(True)
        self.include_tf.set(True)
        self.randomize_questions.set(False)

        # Question type checkboxes
        if mc_count > 0:
            self.create_sleek_checkbox(setup_content, f"Multiple Choice ({mc_count})", self.include_mc)
        if input_count > 0:
            self.create_sleek_checkbox(setup_content, f"Calculations ({input_count})", self.include_input)
        if tf_count > 0:
            self.create_sleek_checkbox(setup_content, f"True/False ({tf_count})", self.include_tf)

        # Separator
        separator = tk.Frame(setup_content, bg=self.colors['border'], height=1)
        separator.pack(fill='x', pady=15)

        options_title = tk.Label(setup_content,
                                 text="Options",
                                 font=('SF Pro Display', 12, 'normal'),
                                 bg=self.colors['card_bg'],
                                 fg=self.colors['text'])
        options_title.pack(anchor='w', pady=(0, 12))

        self.create_sleek_checkbox(setup_content, "Randomize order", self.randomize_questions)

        # Action buttons
        button_container = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_container.pack(fill='x', pady=(20, 0))

        # Button row
        button_row = tk.Frame(button_container, bg=self.colors['card_bg'])
        button_row.pack()

        self.create_sleek_button(button_row, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=5)
        self.create_sleek_button(button_row, "Start Quiz",
                                 lambda: self.start_configured_quiz(quiz_name, available_questions), "primary").pack(
            side='left', padx=5)

    def start_configured_quiz(self, quiz_name, source_questions):
        """Start quiz with configured options"""
        # Filter questions based on selected types
        filtered_questions = []

        for question in source_questions:
            question_type = question.get('type', 'multiple_choice')

            if question_type == 'multiple_choice' and self.include_mc.get():
                filtered_questions.append(question)
            elif question_type == 'input' and self.include_input.get():
                filtered_questions.append(question)
            elif question_type == 'truefalse' and self.include_tf.get():
                filtered_questions.append(question)

        # Check if any questions are selected
        if not filtered_questions:
            self.show_error_popup("Please select at least one question type!")
            return

        # Apply randomization if selected
        self.current_quiz_randomized = self.randomize_questions.get()
        if self.current_quiz_randomized:
            random.shuffle(filtered_questions)

        # Set up the quiz
        self.current_questions = filtered_questions
        self.quiz_name = quiz_name

        # Add configuration info to quiz name
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
        """Show minimal error message"""
        # Create temporary overlay
        overlay = tk.Toplevel(self.root)
        overlay.title("")
        overlay.geometry("350x150")
        overlay.configure(bg=self.colors['card_bg'])
        overlay.resizable(False, False)
        overlay.transient(self.root)
        overlay.grab_set()

        # Center the popup
        overlay.geometry("+%d+%d" % (self.root.winfo_rootx() + 300, self.root.winfo_rooty() + 200))

        # Error content
        error_label = tk.Label(overlay,
                               text=message,
                               font=('SF Pro Display', 11, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['danger'],
                               wraplength=300)
        error_label.pack(expand=True, pady=20)

        # OK button
        ok_btn = self.create_sleek_button(overlay, "OK", overlay.destroy, "primary")
        ok_btn.pack(pady=(0, 20))

    def show_question(self):
        """Display current question with ultra-modern styling"""
        if self.current_question_index >= len(self.current_questions):
            self.show_results()
            return

        self.clear_content()
        question_data = self.current_questions[self.current_question_index]

        # Progress section with circular indicator
        progress_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        progress_container.pack(fill='x', pady=(15, 25))

        # Create circular progress
        self.circular_progress = CircularProgress(progress_container, size=90)

        # Update circular progress
        answered_questions = len(self.user_answers)
        correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
        wrong_count = answered_questions - correct_count

        self.circular_progress.update_progress(len(self.current_questions), correct_count, wrong_count)

        # Progress text below circle
        progress_text = f"Question {self.current_question_index + 1} of {len(self.current_questions)}"
        progress_label = tk.Label(progress_container,
                                  text=progress_text,
                                  font=('Segoe UI', 10, 'normal'),
                                  bg=self.colors['bg'],
                                  fg=self.colors['text_muted'])
        progress_label.pack(pady=(10, 0))

        # Add timer display for Everything Test
        if "Everything Test" in self.quiz_name and self.timer_running:
            self.timer_label = tk.Label(progress_container,
                                        text="⏱️ 00:00",
                                        font=('Segoe UI', 12, 'bold'),
                                        bg=self.colors['bg'],
                                        fg=self.colors['primary'])
            self.timer_label.pack(pady=(5, 0))

        # Question card with minimal styling
        question_card = self.create_ultra_modern_card(self.scrollable_frame, 750)
        question_content = tk.Frame(question_card, bg=self.colors['card_bg'])
        question_content.pack(fill='x', padx=35, pady=25)

        # Question text with perfect typography
        question_label = tk.Label(question_content,
                                  text=question_data['question'],
                                  font=('SF Pro Display', 13, 'normal'),
                                  bg=self.colors['card_bg'],
                                  fg=self.colors['text'],
                                  wraplength=680,
                                  justify='left')
        question_label.pack(pady=(0, 20))

        # Add right-click context menu for copying question text
        self.add_copy_menu(question_label, question_data['question'])

        # Answer input area
        self.setup_answer_input(question_content, question_data)

        # Feedback area (initially hidden)
        self.feedback_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        self.feedback_frame.pack(fill='x', pady=12)

        # Minimal navigation
        nav_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        nav_container.pack(fill='x', pady=18)

        nav_frame = tk.Frame(nav_container, bg=self.colors['bg'])
        nav_frame.pack()

        # Left navigation buttons
        left_nav = tk.Frame(nav_frame, bg=self.colors['bg'])
        left_nav.pack(side='left')

        if self.current_question_index > 0:
            self.create_sleek_button(left_nav, "←", self.previous_question, "ghost").pack(side='left', padx=2)

        self.create_sleek_button(left_nav, "Menu", self.show_main_menu, "ghost").pack(side='left', padx=2)
        self.create_sleek_button(left_nav, "🔄 Restart", self.restart_quiz, "ghost").pack(side='left', padx=2)

        # Submit button
        self.submit_button = self.create_sleek_button(nav_frame, "Submit", self.submit_answer)
        self.submit_button.pack(side='right', padx=3)

    def setup_answer_input(self, parent, question_data):
        """Setup answer input with ultra-modern styling"""
        question_type = question_data.get('type', 'multiple_choice')

        if question_type == 'input':
            # Ultra-clean text input
            input_container = tk.Frame(parent, bg=self.colors['card_bg'])
            input_container.pack(pady=15)

            input_label = tk.Label(input_container,
                                   text="Your answer:",
                                   font=('SF Pro Display', 10, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['text_secondary'])
            input_label.pack(pady=(0, 8))

            # Add copy functionality for input label
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

            # Add copy functionality for input questions
            self.add_copy_menu(input_label, question_data['question'])

            # Also add copy functionality to the input field itself (for entered text)
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
            # Sleek True/False options
            self.selected_answer.set("")

            tf_container = tk.Frame(parent, bg=self.colors['card_bg'])
            tf_container.pack(pady=18)

            tf_frame = tk.Frame(tf_container, bg=self.colors['card_bg'])
            tf_frame.pack()

            self.create_sleek_radiobutton(tf_frame, "TRUE", self.selected_answer, "TRUE")
            self.create_sleek_radiobutton(tf_frame, "FALSE", self.selected_answer, "FALSE")

        else:
            # Ultra-clean multiple choice
            self.selected_answer.set("")

            # Setup shuffled options
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

            # Display options with perfect styling
            options_container = tk.Frame(parent, bg=self.colors['card_bg'])
            options_container.pack(pady=15)

            for letter in option_letters:
                text = self.current_shuffled_options[letter]
                option_text = f"{letter.upper()}. {text}"
                self.create_sleek_radiobutton(options_container, option_text, self.selected_answer, letter)

    def submit_answer(self):
        """Submit answer and show sleek feedback"""
        question_data = self.current_questions[self.current_question_index]
        question_type = question_data.get('type', 'multiple_choice')

        # Get user answer
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

        # Check correctness
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

        # Store answer
        self.user_answers.append({
            'question': question_data['question'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'options': display_options,
            'is_correct': is_correct,
            'type': question_type
        })

        # Update score
        if is_correct:
            self.score += 1

        # Mark question as used
        if question_data not in self.used_questions:
            self.used_questions.append(question_data)

        # Show feedback
        self.show_feedback(is_correct, correct_answer, display_options, question_type)

    def show_feedback(self, is_correct, correct_answer, options, question_type):
        """Show ultra-minimal feedback"""
        self.feedback_visible = True

        self.submit_button.config(text="Continue")

        # ADD THIS LINE TO UPDATE THE CIRCLE:
        # Update circular progress
        answered_questions = len(self.user_answers)
        correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
        wrong_count = answered_questions - correct_count
        self.circular_progress.update_progress(len(self.current_questions), correct_count, wrong_count)

        # Clear previous feedback
        for widget in self.feedback_frame.winfo_children():
            widget.destroy()

        # Create minimal feedback card
        feedback_card = self.create_ultra_modern_card(self.feedback_frame, 650, minimal=True)
        feedback_content = tk.Frame(feedback_card, bg=self.colors['surface'])
        feedback_content.pack(fill='x', padx=20, pady=12)

        # Minimal feedback message
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

        # Add copy functionality to feedback header
        self.add_copy_menu(feedback_header, f"{icon} {message}")

        # Show correct answer if wrong (minimal)
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

            # Add copy functionality to correct answer
            self.add_copy_menu(correct_label, correct_text)

        # Update submit button
        self.submit_button.config(text="Continue")

    def next_question(self):
        """Move to next question"""
        self.feedback_visible = False
        self.current_question_index += 1
        self.show_question()

    def previous_question(self):
        """Go to previous question"""
        if self.current_question_index > 0:
            self.current_question_index -= 1
            if self.user_answers:
                removed_answer = self.user_answers.pop()
                if removed_answer['is_correct']:
                    self.score -= 1
                question_data = self.current_questions[self.current_question_index]
                if question_data in self.used_questions:
                    self.used_questions.remove(question_data)
            self.feedback_visible = False
            self.show_question()

    def show_results(self):
        """Display results with timer and leaderboard option"""
        self.clear_content()

        # Stop timer and get final time
        final_time = self.stop_timer()

        # Force canvas scroll to top
        self.canvas.yview_moveto(0)
        self.root.update_idletasks()

        # Clean results header
        self.create_sleek_title(self.scrollable_frame, "Complete", 26)
        self.create_sleek_subtitle(self.scrollable_frame, f"{self.quiz_name}")

        # Score card with timer
        score_card = self.create_ultra_modern_card(self.scrollable_frame, 500, minimal=True)
        score_content = tk.Frame(score_card, bg=self.colors['surface'])
        score_content.pack(fill='x', padx=30, pady=25)

        percentage = (self.score / len(self.current_questions)) * 100 if len(self.current_questions) > 0 else 0

        # Score display
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

        # Show timer for Everything Test
        if "Everything Test" in self.quiz_name and final_time > 0:
            time_text = tk.Label(score_content,
                                 text=f"⏱️ Time: {self.format_time(final_time)}",
                                 font=('Segoe UI', 14, 'normal'),
                                 bg=self.colors['surface'],
                                 fg=self.colors['warning'])
            time_text.pack(pady=(0, 15))

        # Performance message
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

        # Action buttons
        button_container = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        button_container.pack(fill='x', pady=(30, 20))

        button_frame = tk.Frame(button_container, bg=self.colors['bg'])
        button_frame.pack(anchor='center')

        # Check if eligible for leaderboard submission (Everything Test only)
        if "Everything Test" in self.quiz_name:
            can_submit = self.check_leaderboard_eligibility()

            if can_submit:
                leaderboard_btn = self.create_sleek_button(button_frame, "🏆 Submit Perfect Score",
                                                           lambda: self.submit_to_leaderboard(percentage, final_time),
                                                           "special")
                leaderboard_btn.pack(side='left', padx=8)
            else:
                # Show why they can't submit
                reason = self.get_submission_restriction_reason()
                disabled_btn = self.create_sleek_button(button_frame, f"🚫 {reason}",
                                                        lambda: None, "ghost")
                disabled_btn.config(state='disabled')
                disabled_btn.pack(side='left', padx=8)

            view_leaderboard_btn = self.create_sleek_button(button_frame, "📊 View Leaderboard",
                                                            self.show_leaderboard,
                                                            "secondary")
            view_leaderboard_btn.pack(side='left', padx=8)

        restart_btn = self.create_sleek_button(button_frame, "🔄 Restart", self.restart_quiz, "secondary")
        restart_btn.pack(side='left', padx=8)

        menu_btn = self.create_sleek_button(button_frame, "📋 Menu", self.show_main_menu, "primary")
        menu_btn.pack(side='left', padx=8)

        self.root.update_idletasks()

    def check_leaderboard_eligibility(self):
        """Check if user is eligible to submit to leaderboard"""
        # Must be Everything Test
        if "Everything Test" not in self.quiz_name:
            return False

        # Everything Test is always randomized now, so just check perfect score
        return self.has_perfect_non_input_score()

    def has_perfect_non_input_score(self):
        """Check if user got all non-input questions correct"""
        non_input_questions = [q for q in self.current_questions if q.get('type', 'multiple_choice') != 'input']

        if len(non_input_questions) == 0:
            return False

        # Count correct non-input answers
        correct_non_input = 0
        for i, answer in enumerate(self.user_answers):
            if i < len(self.current_questions):
                question = self.current_questions[i]
                if question.get('type', 'multiple_choice') != 'input' and answer['is_correct']:
                    correct_non_input += 1

        return correct_non_input == len(non_input_questions)

    def get_submission_restriction_reason(self):
        """Get reason why submission is restricted"""
        if not self.has_perfect_non_input_score():
            return "Need Perfect Score"

        return "Not Eligible"


    def show_random_setup(self):
        """Show minimal random quiz setup"""
        self.clear_content()

        self.create_sleek_title(self.scrollable_frame, "Random Quiz", 22)

        # Get available questions
        all_questions = []
        for questions in self.questions.values():
            all_questions.extend(questions)
        available_questions = [q for q in all_questions if q not in self.used_questions]

        if not available_questions:
            # Minimal error message
            error_card = self.create_ultra_modern_card(self.scrollable_frame, 400)
            error_content = tk.Frame(error_card, bg=self.colors['card_bg'])
            error_content.pack(fill='x', padx=25, pady=20)

            error_label = tk.Label(error_content,
                                   text="No questions available",
                                   font=('SF Pro Display', 12, 'normal'),
                                   bg=self.colors['card_bg'],
                                   fg=self.colors['danger'])
            error_label.pack(pady=10)

            self.create_sleek_button(error_content, "← Back", self.show_main_menu, "secondary").pack()
            return

        # Minimal setup card
        setup_card = self.create_ultra_modern_card(self.scrollable_frame, 400)
        setup_content = tk.Frame(setup_card, bg=self.colors['card_bg'])
        setup_content.pack(fill='x', padx=25, pady=20)

        instruction = tk.Label(setup_content,
                               text=f"Questions (max {len(available_questions)})",
                               font=('SF Pro Display', 12, 'normal'),
                               bg=self.colors['card_bg'],
                               fg=self.colors['text'])
        instruction.pack(pady=(0, 12))

        # Minimal number input
        input_frame = tk.Frame(setup_content, bg=self.colors['card_bg'])
        input_frame.pack(pady=8)

        self.random_count_var = tk.StringVar(value=str(min(20, len(available_questions))))
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

        # Minimal buttons
        button_frame = tk.Frame(setup_content, bg=self.colors['card_bg'])
        button_frame.pack(pady=(15, 0))

        self.create_sleek_button(button_frame, "← Back", self.show_main_menu, "ghost").pack(side='left', padx=3)
        self.create_sleek_button(button_frame, "Start", lambda: self.start_random_quiz(available_questions),
                                 "primary").pack(side='left', padx=3)

    def start_random_quiz(self, available_questions):
        """Start random quiz"""
        try:
            count = int(self.random_count_var.get())
            if 1 <= count <= len(available_questions):
                selected_questions = random.sample(available_questions, count)
                self.current_questions = selected_questions
                self.quiz_name = f"Random ({count} questions)"
                self.reset_quiz_state()
                self.show_question()
            else:
                self.show_error_popup(f"Enter 1-{len(available_questions)}")
        except ValueError:
            self.show_error_popup("Enter a valid number")



    def reset_used_questions(self):
        """Reset used questions"""
        self.used_questions = []
        self.show_main_menu()

    def restart_quiz(self):
        """Restart the current quiz from the beginning"""
        # Store original questions before any modifications
        original_questions = self.current_questions.copy()

        # Re-randomize if it was originally randomized
        if self.current_quiz_randomized:
            random.shuffle(original_questions)
            self.current_questions = original_questions

        # Reset state
        self.current_question_index = 0
        self.score = 0
        self.user_answers = []
        self.selected_answer.set("")
        self.feedback_visible = False
        self.current_shuffled_options = {}
        self.current_correct_letter = None

        # Clear any text input
        if hasattr(self, 'text_input'):
            try:
                self.text_input.delete(0, tk.END)
            except:
                pass

        # Restart timer for Everything Test
        if "Everything Test" in self.quiz_name:
            self.start_timer()

        # Show first question again
        self.show_question()

    def reset_quiz_state(self):
        """Reset quiz state"""
        self.current_question_index = 0
        self.score = 0
        self.user_answers = []
        self.selected_answer.set("")
        self.feedback_visible = False
        self.current_shuffled_options = {}
        self.current_correct_letter = None

    def run(self):
        """Start the ultra-modern application"""
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