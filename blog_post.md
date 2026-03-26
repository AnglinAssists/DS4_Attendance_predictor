# I Built a Tool to Predict School Attendance After a Blizzard. Here's What I Learned.

**Mark Anglin** | Senior Literacy Intervention AmeriCorps Member, Reading Partners

---

It was Tuesday morning, February 24th. The Blizzard of 2026 had dropped 22 inches of snow on New York City two days earlier. Schools had been closed on Monday. Now they were open again, and I was standing outside an elementary school in East Harlem wondering the same thing every tutor, teacher, and program coordinator was wondering: how many kids are actually going to show up today?

This is not an idle question for me. I serve as a Senior Literacy Intervention AmeriCorps Member at Reading Partners, working in Community School District 4 in East Harlem, Manhattan. My job is to match struggling readers with trained tutors for one-on-one sessions. When a kid doesn't show up, that session is gone. When half the school doesn't show up, an entire day of programming is disrupted. Tutors are sitting in rooms with no students. Coordinators are scrambling to reschedule. Time that could have been spent on phonics or fluency is just lost.

And nobody could tell me ahead of time how bad it was going to be.

That's the problem. When severe weather shuts schools down and they reopen, programs like ours are flying blind. We staff sessions based on normal attendance. We schedule assessments assuming most kids will be there. We plan family outreach events expecting a crowd. But after a blizzard? We're guessing. And guessing wrong costs real resources and real learning time.

So I decided to stop guessing and start building.

I took 84 days of attendance data from across CSD 4 -- that's 18 elementary schools serving 3,196 students in grades K through 4. I paired each day's attendance numbers with weather data for the same date: temperature, snowfall, wind speed, precipitation. Then I built a prediction tool. Without getting into the technical weeds, what it does is learn the patterns between weather conditions and how many kids come to school. Cold day? A few more absences. Snow on the ground? More absences. Monday after a holiday weekend? Even more. The tool finds these relationships in the historical data and uses them to make a prediction for a new day it hasn't seen before.

I pointed it at February 24th -- the day schools reopened after the blizzard -- and asked it: what's your best guess for attendance?

It said 75%.

On a normal day in CSD 4, attendance runs around 88%. So 75% would mean a serious drop, roughly 415 extra students absent compared to a typical day. That's 415 kids missing lessons, missing meals, missing tutoring sessions.

When the actual numbers came in, the real attendance rate was 79.3%. My prediction was off by about four percentage points. But here's the thing: the tool also produced a confidence range, basically a window that said "I'm fairly sure the real number will fall somewhere between 72% and 78%." The actual result of 79.3% was just above that window. In practical terms, the prediction was in the right neighborhood. It told us to expect a significant drop, and there was a significant drop.

For a first attempt, built by one AmeriCorps member with publicly available data, I'll take it.

But the numbers are only half the story. What matters is what you do with them.

District 4 is a community where 86% of students are economically disadvantaged and 82% are Black or Hispanic. These are families that are disproportionately affected by severe weather. They may not have cars. They may live in walk-up apartments where a snowstorm means carrying a stroller down five flights through uncleared sidewalks. The kids who are absent after a blizzard are not absent because they don't care about school. They're absent because getting there is genuinely hard.

That context makes prediction even more important. If we know attendance will drop, we can plan for it instead of reacting to it.

Here are five things I think districts and programs should consider:

- **Create weather-triggered staffing plans.** If a prediction tool says attendance will be 75%, don't staff tutoring sessions at 100%. Redeploy some of that capacity to phone outreach, makeup session scheduling, or small-group work with the kids who do show up.

- **Prioritize re-engagement on return days.** The day after a closure is when kids are most likely to start a pattern of chronic absence. A quick check-in call or text to families who were absent can make a real difference.

- **Protect meal program resources.** Schools in CSD 4 serve breakfast and lunch to nearly every student. Overproducing food on a low-attendance day is waste. Underproducing means kids who braved the cold don't get fed. Better predictions help cafeterias plan smarter.

- **Reschedule assessments proactively.** Running a standardized reading assessment on a day when a quarter of the class is absent means you'll need to do makeups anyway. If you know attendance will be low, reschedule before the day starts.

- **Use the data to advocate for resources.** When you can show that a blizzard cost your district 415 student-days of instruction in a single morning, that's a concrete number you can bring to budget conversations, grant applications, and conversations with elected officials.

None of this requires a computer science degree. It requires someone who cares enough about the problem to look at the data and ask a simple question: can we do better than guessing?

I think we can.

The full project, including the code, the data, and a detailed writeup of the methodology, is available on GitHub: [github.com/AnglinAssists/DS4_Attendance_predictor](https://github.com/AnglinAssists/DS4_Attendance_predictor). Everything is open source. If you work in a school district or education nonprofit and want to adapt this for your community, reach out.

I'm currently exploring roles where data meets people -- positions in program management, community impact, or applied research where I can use tools like this to make organizations smarter about the communities they serve. If that sounds like your team, I'd love to connect.

The next blizzard is coming. The question is whether we'll be ready for it.

---

*Mark Anglin is a Senior Literacy Intervention AmeriCorps Member at Reading Partners, serving elementary schools in East Harlem. He holds a background in data analysis and community-based education.*
