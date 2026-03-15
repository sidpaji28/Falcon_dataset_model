export default function SuperAdminDashboard() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">Super Admin Dashboard</h1>
      <p>System oversight, user management, and API request volume metrics. Monitor background job workers.</p>
      <div className="mt-4 p-4 border rounded">
        <h2 className="text-xl font-semibold">Agent Monitoring</h2>
        <p className="mt-2 text-gray-700">Currently running 12-hour background cycles: <span className="text-green-600 font-bold">Active</span></p>
      </div>
    </div>
  );
}
